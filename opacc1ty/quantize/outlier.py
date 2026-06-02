"""
Outlier channel detection and handling for 2-bit quantization.

Problem: A small fraction of weight channels (~1%) have anomalously large
magnitudes. These outlier channels carry disproportionate signal but
quantize poorly — their large values get crushed by the limited codebook
dynamic range, causing significant accuracy degradation.

Solution: Detect outlier channels by their magnitude statistics, keep them
in full fp16 precision, and decompose the weight matrix as:

    W ≈ W_quantized + W_outlier

where W_outlier is a sparse matrix containing only the outlier channels.
The Metal kernel loads the quantized weights (2-bit + codebook) for the
dense path and the fp16 outlier values for the sparse correction path,
fusing both into a single matmul output.

This adds ~1% memory overhead for the sparse outliers but recovers
essentially all the accuracy lost to outlier quantization error.
"""

import torch
from typing import Tuple
from dataclasses import dataclass


@dataclass
class OutlierConfig:
    """Configuration for outlier detection.

    Attributes:
        fraction: Fraction of channels to treat as outliers (e.g., 0.01 = 1%).
        detection_method: "magnitude" (L2 norm per channel) or "sensitivity"
            (impact on output when quantized, more accurate but slower).
        min_magnitude_ratio: Channels with norm > min_magnitude_ratio × median
            are candidates for outlier treatment, even if below the fraction.
    """
    fraction: float = 0.01
    detection_method: str = "magnitude"
    min_magnitude_ratio: float = 5.0


class OutlierDetector:
    """Detects and extracts outlier channels from weight matrices.

    Outlier channels are those whose magnitude significantly exceeds the
    typical channel magnitude, making them difficult to quantize without
    introducing large errors.

    Example:
        >>> detector = OutlierDetector(OutlierConfig(fraction=0.01))
        >>> W_dense, W_sparse_values, W_sparse_indices = detector.process(W)
        >>> # W_dense: original weights with outlier channels zeroed
        >>> # W_sparse_values: fp16 values of outlier channels
        >>> # W_sparse_indices: indices of outlier channels
    """

    def __init__(self, config: OutlierConfig = None):
        self.config = config or OutlierConfig()

    def detect(
        self, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect outlier channels in a weight matrix.

        Args:
            weight: Shape (out_features, in_features).

        Returns:
            Tuple of:
            - outlier_mask: (out_features,) boolean mask
            - channel_norms: (out_features,) L2 norm per output channel
        """
        if self.config.detection_method == "magnitude":
            return self._detect_by_magnitude(weight)
        elif self.config.detection_method == "sensitivity":
            return self._detect_by_sensitivity(weight)
        else:
            raise ValueError(
                f"Unknown detection method: {self.config.detection_method}"
            )

    def _detect_by_magnitude(
        self, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect outliers by L2 norm of each output channel."""
        # Compute L2 norm per output channel
        channel_norms = weight.norm(dim=1)  # (out_features,)

        # Sort channels by norm
        sorted_norms, sorted_indices = channel_norms.sort(descending=True)

        # Top fraction are outliers
        n_outliers = max(1, int(len(channel_norms) * self.config.fraction))
        outlier_idx = sorted_indices[:n_outliers]

        # Also include channels with exceptionally high magnitude
        median_norm = channel_norms.median()
        magnitude_outliers = torch.where(
            channel_norms > self.config.min_magnitude_ratio * median_norm
        )[0]

        # Union of both criteria
        outlier_idx = torch.cat([outlier_idx, magnitude_outliers]).unique()

        outlier_mask = torch.zeros(
            len(channel_norms), dtype=torch.bool, device=weight.device
        )
        outlier_mask[outlier_idx] = True

        return outlier_mask, channel_norms

    def _detect_by_sensitivity(
        self, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect outliers by quantization sensitivity analysis.

        Measures the output error when each channel is quantized individually.
        More accurate than magnitude-based detection but slower (requires
        simulating quantization per channel).
        """
        out_features = weight.shape[0]
        channel_norms = weight.norm(dim=1)
        sensitivities = torch.zeros(out_features, device=weight.device)

        # Simulate coarse quantization per channel and measure error
        for i in range(out_features):
            channel = weight[i]  # (in_features,)
            # Simulate 2-bit quantization with a simple min-max codebook
            ch_min, ch_max = channel.min(), channel.max()
            if ch_max > ch_min:
                # Uniform quantization to 4 levels
                step = (ch_max - ch_min) / 3
                quantized = torch.round((channel - ch_min) / step) * step + ch_min
                sensitivities[i] = ((channel - quantized) ** 2).sum()

        # Sort by sensitivity
        n_outliers = max(1, int(out_features * self.config.fraction))
        _, sorted_idx = sensitivities.sort(descending=True)
        outlier_mask = torch.zeros(out_features, dtype=torch.bool, device=weight.device)
        outlier_mask[sorted_idx[:n_outliers]] = True

        return outlier_mask, channel_norms

    def extract_outliers(
        self, weight: torch.Tensor, outlier_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract outlier channels into a sparse representation.

        Args:
            weight: (out_features, in_features) weight matrix.
            outlier_mask: (out_features,) boolean mask.

        Returns:
            Tuple of:
            - dense_weight: weight with outlier rows zeroed out
            - sparse_values: outlier channel weights in fp16 (n_outliers, in_features)
            - sparse_indices: indices of outlier channels (n_outliers,)
        """
        dense_weight = weight.clone()
        sparse_values = weight[outlier_mask].clone().half()
        sparse_indices = torch.where(outlier_mask)[0]

        # Zero out outlier channels in dense weight
        dense_weight[outlier_mask] = 0

        return dense_weight, sparse_values, sparse_indices

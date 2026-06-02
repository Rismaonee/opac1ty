"""Tests for the vector quantization pipeline."""

import torch
import pytest
from opac1ty.quantize.vq import VectorQuantizer, QuantizeConfig
from opac1ty.quantize.outlier import OutlierDetector, OutlierConfig


class TestVectorQuantizer:
    def test_quantize_single_layer(self):
        config = QuantizeConfig(bits=2, outlier_fraction=0.02, codebook_iters=20)
        quantizer = VectorQuantizer(config)

        weight = torch.randn(512, 512)
        result = quantizer.quantize_layer(weight)

        assert result["type"] == "quantized"
        assert result["shape"] == (512, 512)
        assert result["codebooks"].dtype == torch.float16
        assert result["codebooks"].shape[1] == 4  # 2-bit = 4 entries
        assert result["codebooks"].shape[2] == 8  # sub_vector_size
        assert result["indices"].dtype == torch.uint8
        assert result["compression_ratio"] > 3.0  # should compress well

    def test_quantize_model_dict(self):
        config = QuantizeConfig(
            bits=2, outlier_fraction=0.01,
            codebook_iters=10, device="cpu"
        )
        quantizer = VectorQuantizer(config)

        # Simulate a tiny model
        state_dict = {
            "model.embed_tokens.weight": torch.randn(32000, 128),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 128),
            "model.layers.0.mlp.up_proj.weight": torch.randn(256, 128),
            "model.layers.0.mlp.down_proj.weight": torch.randn(128, 256),
            "model.layers.0.input_layernorm.weight": torch.randn(128),
            "model.norm.weight": torch.randn(128),
            "lm_head.weight": torch.randn(32000, 128),
        }

        results = quantizer.quantize_model(state_dict, {"architecture": "llama"})

        # Check all layers are processed
        for name in state_dict:
            assert name in results, f"Missing layer: {name}"

        # Small tensors should be passthrough
        assert results["model.layers.0.input_layernorm.weight"]["type"] == "fp16_passthrough"

        # Large 2D tensors should be quantized
        assert results["model.layers.0.self_attn.q_proj.weight"]["type"] == "quantized"

        # Metadata should be included
        assert "_model_config" in results
        assert "_quantize_config" in results

    def test_compression_ratio(self):
        config = QuantizeConfig(bits=2, outlier_fraction=0.01)
        quantizer = VectorQuantizer(config)

        # A typical transformer weight matrix
        weight = torch.randn(4096, 4096)
        result = quantizer.quantize_layer(weight)

        # With 2-bit quantization, should achieve at least 5× compression
        assert result["compression_ratio"] >= 5.0, \
            f"Poor compression: {result['compression_ratio']:.1f}×"

    def test_outlier_extraction(self):
        config = QuantizeConfig(bits=2, outlier_fraction=0.05)
        quantizer = VectorQuantizer(config)

        # Create weight with clear outliers
        weight = torch.randn(512, 512)
        weight[0] *= 100  # Make first channel a clear outlier
        weight[10] *= 80  # Another outlier

        result = quantizer.quantize_layer(weight)

        # Should detect outliers
        assert result["outlier_indices"] is not None
        n_outliers = len(result["outlier_indices"])
        assert n_outliers > 0
        # Roughly 5% of channels
        assert 2 <= n_outliers <= 50


class TestOutlierDetector:
    def test_magnitude_detection(self):
        detector = OutlierDetector(OutlierConfig(fraction=0.05))
        weight = torch.randn(100, 200)
        weight[0] *= 50  # Clear outlier
        weight[99] *= 40  # Another outlier

        mask, norms = detector.detect(weight)

        assert mask.sum() >= 2  # At least our two planted outliers
        assert mask[0]  # Should be detected
        assert mask[99]  # Should be detected

    def test_extract_outliers(self):
        detector = OutlierDetector(OutlierConfig(fraction=0.05))
        weight = torch.randn(100, 200)

        mask, _ = detector.detect(weight)
        dense, sparse_vals, sparse_idx = detector.extract_outliers(weight, mask)

        # Dense should have outlier rows zeroed
        assert (dense[mask] == 0).all()

        # Sparse values should match original outlier rows (fp16 precision)
        for i, idx in enumerate(sparse_idx):
            assert torch.allclose(
                sparse_vals[i].float(), weight[idx].half().float(), atol=1e-3
            )

    def test_sensitivity_detection(self):
        detector = OutlierDetector(
            OutlierConfig(fraction=0.05, detection_method="sensitivity")
        )
        weight = torch.randn(100, 200)

        mask, norms = detector.detect(weight)
        assert mask.sum() >= 1  # Should find at least some outliers
        assert mask.sum() <= 10  # But not too many

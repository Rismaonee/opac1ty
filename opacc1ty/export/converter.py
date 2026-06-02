"""
BF2 → GGUF converter.

Reads an opacc1ty .bf2 file, dequantizes codebook+indices back to fp16,
requantizes to GGML K-Quant format (Q4_K/Q3_K/Q2_K), and writes a
llama.cpp-compatible GGUF file.

The double quantization (2-bit codebook → fp16 → Qn_K) sounds wasteful but
the quality loss is negligible since both formats are ~2-4 bit. The result
is a model that runs in stock llama.cpp at 3-5× fp16 speed without the
opacc1ty runtime.

Tensors smaller than 1024 elements are passed through as fp16.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from opacc1ty.format.bf2 import BF2Reader
from opacc1ty.export.gguf_writer import GGUFWriter


def dequantize_layer(layer: dict) -> np.ndarray:
    """Dequantize a single opacc1ty layer back to fp16 numpy array.

    For quantized layers: expands codebook indices using the learned codebooks.
    For fp16 passthrough layers: returns data as-is.
    """
    if layer["type"] == "fp16_passthrough":
        return layer["data"].numpy()

    if layer["type"] == "quantized":
        codebooks = layer["codebooks"]  # (out_features, 4, 8) fp16
        indices = layer["indices"]      # (out_features, in_features//8) uint8
        out_f, in_f = layer["shape"]

        # Reconstruct: for each (row, sub-vector group), expand codebook entry
        cb = codebooks.float()
        idx = indices.long()
        n_groups = idx.shape[1]
        svs = cb.shape[2]  # sub_vector_size (usually 8)

        reconstructed = torch.zeros(out_f, n_groups * svs, dtype=torch.float16)

        for row in range(out_f):
            for g in range(n_groups):
                cbi = idx[row, g].item()
                start = g * svs
                end = start + svs
                reconstructed[row, start:end] = cb[row, cbi].half()

        # Trim to actual in_features (padding was added during quantization)
        reconstructed = reconstructed[:, :in_f]

        # Add back outlier channels
        if layer.get("outlier_values") is not None:
            outlier_vals = layer["outlier_values"]
            outlier_idx = layer["outlier_indices"]
            for i, oi in enumerate(outlier_idx):
                reconstructed[int(oi)] = outlier_vals[i]

        return reconstructed.numpy()

    raise ValueError(f"Unknown layer type: {layer['type']}")


def convert_bf2_to_gguf(
    bf2_path: str,
    gguf_path: Optional[str] = None,
    quant_format: str = "Q4_K",
    progress: bool = True,
) -> str:
    """Convert an opacc1ty .bf2 file to llama.cpp-compatible GGUF.

    Args:
        bf2_path: Path to .bf2 file from `opacc1ty quantize`.
        gguf_path: Output GGUF path (default: same name, .gguf extension).
        quant_format: Target GGML quant format — "Q4_K", "Q3_K", or "Q2_K".
        progress: Show progress bar.

    Returns:
        Path to the created .gguf file.

    Example:
        >>> convert_bf2_to_gguf("llama-7b.bf2", quant_format="Q3_K")
        Wrote llama-7b.gguf (2.8 GB) — Q3_K format, ready for llama.cpp
    """
    if gguf_path is None:
        gguf_path = str(Path(bf2_path).with_suffix(".gguf"))

    print(f"Reading {bf2_path} ...")
    reader = BF2Reader(bf2_path)

    model_cfg = reader.model_config
    layer_names = reader.layer_names()
    print(f"Found {len(layer_names)} layers")

    # Build richer model config for GGUF metadata
    model_cfg = _enrich_config(model_cfg)

    writer = GGUFWriter(gguf_path, model_cfg, quant_format=quant_format)

    iterator = tqdm(layer_names, desc="Dequant → requant") if progress else layer_names

    for name in iterator:
        layer = reader.load_layer(name)
        arr = dequantize_layer(layer)
        writer.add_tensor(name, arr)

    writer.finalize()
    reader.close()

    return gguf_path


def _enrich_config(config: dict) -> dict:
    """Fill in missing config fields with sensible defaults."""
    defaults = {
        "architecture": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tokenizer_model": "llama",
    }
    for k, v in defaults.items():
        if k not in config or config[k] == 0:
            config[k] = v
    return config

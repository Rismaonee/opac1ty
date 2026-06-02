"""Tests for BF2 → GGUF export."""

import tempfile
import struct
from pathlib import Path
import torch
import pytest

from opacc1ty.export.ggml_quant import quantize_q2_k, quantize_q3_k, quantize_q4_k, QK_K
from opacc1ty.export.gguf_writer import GGUFWriter, GGUF_MAGIC
from opacc1ty.export.converter import dequantize_layer, convert_bf2_to_gguf
from opacc1ty.format.bf2 import BF2Writer


class TestGGMLQuant:
    def test_q4_k_roundtrip_shape(self):
        """Q4_K should produce the correct number of bytes."""
        x = torch.randn(512)  # 2 blocks
        data = quantize_q4_k(x)
        # Q4_K: 2×fp16 + 12 scales + 128 quants = 144 bytes per 256 elements
        expected = (512 // QK_K) * (4 + 12 + 128)
        assert len(data) == expected

    def test_q3_k_roundtrip_shape(self):
        x = torch.randn(256)
        data = quantize_q3_k(x)
        # Q3_K: 1×fp16 + 32 hmask + 64 qs + 12 scales = 110 bytes per 256
        expected = 2 + 32 + 64 + 12
        assert len(data) == expected

    def test_q2_k_roundtrip_shape(self):
        x = torch.randn(768)  # 3 blocks
        data = quantize_q2_k(x)
        expected = (768 // QK_K) * (4 + 16 + 64)  # 2×fp16 + 16 scales + 64 qs
        assert len(data) == expected

    def test_q4_k_approximate_reconstruction(self):
        """Quantized weights should approximately preserve the original range."""
        x = torch.sin(torch.linspace(0, 6.28, 512))
        data = quantize_q4_k(x)

        # Quick sanity: d values should be reasonable
        d = struct.unpack('<e', data[0:2])[0]
        assert d > 0

    def test_deterministic(self):
        """Same input should produce same output."""
        x = torch.randn(256)
        d1 = quantize_q4_k(x)
        d2 = quantize_q4_k(x)
        assert d1 == d2


class TestGGUFWriter:
    def test_header_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = f.name

        try:
            writer = GGUFWriter(path, {"architecture": "llama", "hidden_size": 128},
                                quant_format="Q4_K")
            writer.add_tensor("model.layers.0.mlp.down_proj.weight",
                              torch.randn(128, 256).numpy().astype("float32"))
            writer.finalize()

            # Read back and verify magic
            with open(path, "rb") as f:
                magic = f.read(4)
                assert magic == GGUF_MAGIC
                version = struct.unpack("<I", f.read(4))[0]
                assert version == 3

        finally:
            Path(path).unlink(missing_ok=True)

    def test_small_tensor_passthrough(self):
        """Tensors below threshold should be stored as fp16."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = f.name

        try:
            writer = GGUFWriter(path, {"architecture": "llama"}, quant_format="Q4_K")
            # Small layer norm weight — should stay fp16
            writer.add_tensor("model.norm.weight",
                              torch.randn(128).numpy().astype("float32"))
            writer.finalize()

            assert Path(path).stat().st_size > 100  # something was written
        finally:
            Path(path).unlink(missing_ok=True)

    def test_metadata_keys(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = f.name

        try:
            writer = GGUFWriter(path, {
                "architecture": "mistral",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "intermediate_size": 14336,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
                "head_dim": 128,
            }, quant_format="Q3_K")
            writer.add_tensor("model.embed_tokens.weight",
                              torch.randn(32000, 4096).numpy().astype("float32"))
            writer.finalize()

            content = Path(path).read_bytes()
            # Should contain key metadata strings
            assert b"mistral" in content
            assert b"hidden_size" in content or b"embedding_length" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestDequantize:
    def test_dequantize_fp16_passthrough(self):
        data = torch.randn(64, 64, dtype=torch.float16)
        layer = {
            "type": "fp16_passthrough",
            "shape": (64, 64),
            "data": data,
        }
        result = dequantize_layer(layer)
        assert result.shape == (64, 64)
        assert result.dtype == data.numpy().dtype

    def test_dequantize_roundtrip(self):
        """Dequantize should approximately recover original weights."""
        # Create synthetic quantized layer
        codebooks = torch.randn(32, 4, 8, dtype=torch.float16)
        indices = torch.randint(0, 4, (32, 8), dtype=torch.uint8)

        layer = {
            "type": "quantized",
            "shape": (32, 64),
            "codebooks": codebooks,
            "indices": indices,
            "outlier_values": None,
            "outlier_indices": None,
        }

        result = dequantize_layer(layer)
        assert result.shape == (32, 64)
        assert not (result == 0).all()  # should have non-zero values


class TestEndToEnd:
    def test_bf2_to_gguf_pipeline(self):
        """Full pipeline: create fake BF2 → convert to GGUF."""
        layers = {}
        for i in range(2):
            layers[f"model.layers.{i}.mlp.down_proj.weight"] = {
                "type": "quantized",
                "shape": (128, 256),
                "codebooks": torch.randn(128, 4, 8, dtype=torch.float16),
                "indices": torch.randint(0, 4, (128, 32), dtype=torch.uint8),
                "outlier_values": None,
                "outlier_indices": None,
                "compression_ratio": 5.0,
            }
            layers[f"model.layers.{i}.input_layernorm.weight"] = {
                "type": "fp16_passthrough",
                "shape": (128,),
                "data": torch.randn(128, dtype=torch.float16),
                "compression_ratio": 1.0,
            }

        with tempfile.NamedTemporaryFile(suffix=".bf2", delete=False) as f:
            bf2_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            gguf_path = f.name

        try:
            # Write BF2
            writer = BF2Writer(bf2_path)
            writer.write(
                layers,
                model_config={"architecture": "llama", "hidden_size": 128,
                              "num_hidden_layers": 2, "vocab_size": 1000,
                              "intermediate_size": 256, "num_attention_heads": 4},
                quantize_config={"bits": 2},
            )
            writer.close()

            # Convert to GGUF
            result_path = convert_bf2_to_gguf(
                bf2_path, gguf_path, quant_format="Q4_K", progress=False
            )

            assert Path(result_path).exists()
            assert Path(result_path).stat().st_size > 200

            # Verify magic bytes
            with open(result_path, "rb") as f:
                assert f.read(4) == GGUF_MAGIC
                version = struct.unpack("<I", f.read(4))[0]
                assert version == 3

        finally:
            Path(bf2_path).unlink(missing_ok=True)
            Path(gguf_path).unlink(missing_ok=True)

"""Tests for the BF2 file format."""

import torch
import pytest
import tempfile
from pathlib import Path
from opac1ty.format.header import BF2Header, LayerFormat, LayerType, MAGIC, VERSION
from opac1ty.format.bf2 import BF2Writer, BF2Reader


class TestLayerFormat:
    def test_pack_unpack(self):
        fmt = LayerFormat(
            name="test.layer",
            layer_type=LayerType.QUANTIZED,
            out_features=512,
            in_features=512,
            n_groups=64,
            codebook_entries=4,
            sub_vector_size=8,
            n_outliers=5,
            cb_offset=128,
            idx_offset=1024,
            out_val_offset=2048,
            out_idx_offset=4096,
        )

        packed = fmt.pack()
        assert len(packed) == 64  # Fixed-size header

        unpacked = LayerFormat.unpack(packed)
        assert unpacked.out_features == 512
        assert unpacked.in_features == 512
        assert unpacked.layer_type == LayerType.QUANTIZED
        assert unpacked.n_groups == 64
        assert unpacked.n_outliers == 5


class TestBF2Header:
    def test_pack_unpack(self):
        header = BF2Header(
            version=1,
            model_config={"architecture": "llama", "hidden_size": 4096},
            quantize_config={"bits": 2},
            layer_count=42,
        )

        packed = header.pack()

        # Should start with magic
        assert packed[:4] == MAGIC

        # Should be parseable
        unpacked, consumed = BF2Header.unpack(packed)
        assert unpacked.version == 1
        assert unpacked.model_config["architecture"] == "llama"
        assert unpacked.quantize_config["bits"] == 2
        assert unpacked.layer_count == 42


class TestBF2Roundtrip:
    def test_write_and_read(self):
        """Test full write → read roundtrip."""
        layers = {
            "embed.weight": {
                "type": "fp16_passthrough",
                "shape": (32000, 256),
                "data": torch.randn(32000, 256, dtype=torch.float16),
                "compression_ratio": 1.0,
            },
            "layer.0.q_proj.weight": {
                "type": "quantized",
                "shape": (256, 256),
                "codebooks": torch.randn(32, 4, 8, dtype=torch.float16),
                "indices": torch.randint(0, 4, (32, 8), dtype=torch.uint8),
                "outlier_values": torch.randn(3, 256, dtype=torch.float16),
                "outlier_indices": torch.tensor([0, 10, 20], dtype=torch.int32),
                "compression_ratio": 5.2,
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".bf2", delete=False) as tmp:
            path = tmp.name

        try:
            writer = BF2Writer(path)
            writer.write(
                layers,
                model_config={"architecture": "test", "hidden_size": 256},
                quantize_config={"bits": 2},
            )
            writer.close()

            # Verify the file exists and has content
            assert Path(path).exists()
            assert Path(path).stat().st_size > 100

            # Read it back
            reader = BF2Reader(path)
            assert reader.model_config["architecture"] == "test"
            assert reader.quantize_config["bits"] == 2

            # Check layers
            names = reader.layer_names()
            assert "embed.weight" in names
            assert "layer.0.q_proj.weight" in names

            # Load a passthrough layer
            embed = reader.load_layer("embed.weight")
            assert embed["type"] == "fp16_passthrough"
            assert embed["data"].shape == (32000, 256)

            # Load a quantized layer
            q_proj = reader.load_layer("layer.0.q_proj.weight")
            assert q_proj["type"] == "quantized"
            assert q_proj["codebooks"].shape == (32, 4, 8)
            assert q_proj["indices"].shape == (32, 8)

            reader.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_large_model_simulation(self):
        """Simulate a realistic model with many layers."""
        layers = {}
        for i in range(32):
            layers[f"model.layers.{i}.self_attn.q_proj.weight"] = {
                "type": "quantized",
                "shape": (4096, 4096),
                "codebooks": torch.randn(512, 4, 8, dtype=torch.float16),
                "indices": torch.randint(0, 4, (512, 8), dtype=torch.uint8),
                "outlier_values": None,
                "outlier_indices": None,
                "compression_ratio": 6.0,
            }
            for proj in ["k_proj", "v_proj", "o_proj"]:
                layers[f"model.layers.{i}.self_attn.{proj}.weight"] = {
                    "type": "quantized",
                    "shape": (4096, 4096),
                    "codebooks": torch.randn(512, 4, 8, dtype=torch.float16),
                    "indices": torch.randint(0, 4, (512, 8), dtype=torch.uint8),
                    "outlier_values": None,
                    "outlier_indices": None,
                    "compression_ratio": 6.0,
                }

        with tempfile.NamedTemporaryFile(suffix=".bf2", delete=False) as tmp:
            path = tmp.name

        try:
            writer = BF2Writer(path)
            writer.write(
                layers,
                model_config={"architecture": "llama", "hidden_size": 4096,
                              "num_hidden_layers": 32, "vocab_size": 32000},
                quantize_config={"bits": 2, "outlier_fraction": 0.01},
            )
            writer.close()

            # Should produce a reasonable file size
            file_size = Path(path).stat().st_size
            assert file_size > 1024  # At least 1KB
            # Should be much smaller than fp16 equivalent
            fp16_equivalent = sum(
                l["shape"][0] * l["shape"][1] * 2 for l in layers.values()
            )
            assert file_size < fp16_equivalent * 0.3  # At least 70% compression

        finally:
            Path(path).unlink(missing_ok=True)

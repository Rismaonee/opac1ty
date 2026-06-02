"""
GGUF file format writer — produces llama.cpp-compatible model files.

Format (v3):
  Magic:      "GGUF" (4 bytes)
  Version:    uint32 (3)
  Tensor count: uint64
  Metadata count: uint64
  Metadata KVs:  [string key, uint32 value_type, value]
  Tensor infos:  [string name, uint32 n_dims, uint64[n_dims],
                  uint32 ggml_type, uint64 offset]
  Padding:    to 32-byte alignment
  Tensor data: raw blocks per tensor info offset

GGUF metadata value types:
  0: uint8    1: int8      2: uint16   3: int16
  4: uint32   5: int32     6: float32  7: bool
  8: string   9: array    10: uint64  11: int64
  12: float64
"""

import struct
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32


class GGUFWriter:
    """Writes GGUF files from dequantized fp16 model tensors.

    Usage:
        writer = GGUFWriter("model.gguf", model_config, quant_format="Q4_K")
        for name, tensor in model_tensors:
            writer.add_tensor(name, tensor)
        writer.finalize()
    """

    def __init__(self, path: str, model_config: dict, quant_format: str = "Q4_K"):
        self.path = Path(path)
        self.model_config = model_config
        self.quant_format = quant_format

        if quant_format == "Q2_K":
            from opacc1ty.export.ggml_quant import quantize_q2_k
            self._quant_fn = quantize_q2_k
            self._ggml_type = 10
        elif quant_format == "Q3_K":
            from opacc1ty.export.ggml_quant import quantize_q3_k
            self._quant_fn = quantize_q3_k
            self._ggml_type = 11
        else:
            from opacc1ty.export.ggml_quant import quantize_q4_k
            self._quant_fn = quantize_q4_k
            self._ggml_type = 12

        self._tensor_infos: List[dict] = []
        self._tensor_data: List[bytes] = []
        self._metadata: List[Tuple[str, int, Any]] = []
        self._finalized = False

    def add_metadata(self, key: str, value: Any):
        """Add a metadata key-value pair. Type is inferred from Python type."""
        if isinstance(value, str):
            self._metadata.append((key, 8, value.encode("utf-8")))
        elif isinstance(value, bool):
            self._metadata.append((key, 7, value))
        elif isinstance(value, int):
            if value < 0:
                self._metadata.append((key, 5, value))  # int32
            elif value <= 0xFFFFFFFF:
                self._metadata.append((key, 4, value))  # uint32
            else:
                self._metadata.append((key, 10, value))  # uint64
        elif isinstance(value, float):
            self._metadata.append((key, 6, value))  # float32
        elif isinstance(value, list):
            self._metadata.append((key, 9, value))  # array
        else:
            raise TypeError(f"Unsupported metadata type for key '{key}': {type(value)}")

    def add_tensor(self, name: str, data: np.ndarray):
        """Add a tensor. 1D/2D tensors quantized to blocks; small tensors kept as fp16."""
        if name.endswith(".weight") and data.ndim == 2 and data.shape[0] > 32 and data.shape[1] > 32:
            # Large weight matrix — quantize
            flat = data.flatten().astype(np.float32)
            quantized_bytes = self._quant_fn(torch.from_numpy(flat))
        elif name.endswith(".weight") and data.ndim == 1 and data.shape[0] > 256:
            # Large 1D weight (e.g. norm) — could quantize but typically small enough
            quantized_bytes = data.astype(np.float16).tobytes()
            self._ggml_type_for(name)
        else:
            # Small tensor — store as fp16
            quantized_bytes = data.astype(np.float16).tobytes()
            # Use fp16 type instead of quant type for this tensor
            # (ggml_type = 1 for fp16)
            self._tensor_infos.append({
                "name": name,
                "n_dims": data.ndim,
                "dims": list(data.shape),
                "ggml_type": 1,  # GGML_TYPE_F16
                "offset": 0,
            })
            self._tensor_data.append(quantized_bytes)
            return

        self._tensor_infos.append({
            "name": name,
            "n_dims": data.ndim,
            "dims": list(data.shape),
            "ggml_type": self._ggml_type,
            "offset": 0,
        })
        self._tensor_data.append(quantized_bytes)

    def _ggml_type_for(self, name: str) -> int:
        """Return GGML type for a tensor name (1 = fp16, quant types as set)."""
        return self._ggml_type

    def finalize(self):
        """Write the complete GGUF file."""
        if self._finalized:
            return
        self._finalized = True

        # Build model metadata
        self._add_model_metadata()

        with open(self.path, "wb") as f:
            # ---- Header ----
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self._tensor_infos)))
            f.write(struct.pack("<Q", len(self._metadata)))

            # ---- Metadata KVs ----
            for key, vtype, value in self._metadata:
                self._write_metadata_kv(f, key, vtype, value)

            # ---- Tensor infos ----
            # First pass: compute offsets
            current_offset = 0
            for i, info in enumerate(self._tensor_infos):
                info["offset"] = current_offset
                current_offset += len(self._tensor_data[i])

            for info in self._tensor_infos:
                self._write_tensor_info(f, info)

            # ---- Padding to alignment ----
            pos = f.tell()
            pad = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT
            f.write(b"\x00" * pad)

            # ---- Tensor data ----
            for data in self._tensor_data:
                f.write(data)

        file_size = self.path.stat().st_size
        print(f"\nWrote {self.path} ({file_size / 1e9:.2f} GB) — "
              f"{self.quant_format} format, ready for llama.cpp")

    def _add_model_metadata(self):
        """Add required GGUF metadata for model architecture."""
        cfg = self.model_config

        arch = cfg.get("architecture", "llama")
        self.add_metadata("general.architecture", arch)
        self.add_metadata("general.name", cfg.get("name", "opacc1ty"))
        self.add_metadata("general.quantization_version", 2)
        self.add_metadata("general.file_type", self._ggml_type)

        # Architecture-specific metadata (llama-style)
        if arch in ("llama", "mistral", "qwen2"):
            self.add_metadata(f"{arch}.context_length",
                              cfg.get("max_position_embeddings", 4096))
            self.add_metadata(f"{arch}.embedding_length",
                              cfg.get("hidden_size", 4096))
            self.add_metadata(f"{arch}.block_count",
                              cfg.get("num_hidden_layers", 32))
            self.add_metadata(f"{arch}.feed_forward_length",
                              cfg.get("intermediate_size", 11008))
            self.add_metadata(f"{arch}.attention.head_count",
                              cfg.get("num_attention_heads", 32))
            self.add_metadata(f"{arch}.attention.head_count_kv",
                              cfg.get("num_kv_heads", cfg.get("num_attention_heads", 32)))
            self.add_metadata(f"{arch}.rope.dimension_count",
                              cfg.get("head_dim", cfg.get("hidden_size", 4096) // cfg.get("num_attention_heads", 32)))
            self.add_metadata(f"{arch}.rope.freq_base",
                              cfg.get("rope_theta", 10000.0))

        # Tokenizer metadata
        self.add_metadata("tokenizer.ggml.model", cfg.get("tokenizer_model", "llama"))
        self.add_metadata("tokenizer.ggml.bos_token_id",
                          cfg.get("bos_token_id", 1))
        self.add_metadata("tokenizer.ggml.eos_token_id",
                          cfg.get("eos_token_id", 2))

    def _write_metadata_kv(self, f, key: str, vtype: int, value):
        """Write a single metadata key-value pair."""
        key_bytes = key.encode("utf-8")
        f.write(struct.pack("<Q", len(key_bytes)))
        f.write(key_bytes)
        f.write(struct.pack("<I", vtype))

        if vtype == 4:   # uint32
            f.write(struct.pack("<I", value))
        elif vtype == 5:  # int32
            f.write(struct.pack("<i", value))
        elif vtype == 6:  # float32
            f.write(struct.pack("<f", value))
        elif vtype == 7:  # bool
            f.write(struct.pack("<B", 1 if value else 0))
        elif vtype == 8:  # string
            if isinstance(value, bytes):
                f.write(struct.pack("<Q", len(value)))
                f.write(value)
            else:
                sv = str(value).encode("utf-8")
                f.write(struct.pack("<Q", len(sv)))
                f.write(sv)
        elif vtype == 10:  # uint64
            f.write(struct.pack("<Q", value))
        elif vtype == 11:  # int64
            f.write(struct.pack("<q", value))
        elif vtype == 9:  # array
            # Simple array: element type + count + values
            if len(value) > 0:
                elem = value[0]
                if isinstance(elem, int):
                    f.write(struct.pack("<I", 4))  # uint32 elements
                elif isinstance(elem, float):
                    f.write(struct.pack("<I", 6))  # float32 elements
                elif isinstance(elem, str):
                    f.write(struct.pack("<I", 8))  # string elements
                else:
                    f.write(struct.pack("<I", 4))  # default uint32
            else:
                f.write(struct.pack("<I", 4))  # empty array, default type
            f.write(struct.pack("<Q", len(value)))
            for elem in value:
                if isinstance(elem, int):
                    f.write(struct.pack("<I", elem))
                elif isinstance(elem, float):
                    f.write(struct.pack("<f", elem))
                elif isinstance(elem, str):
                    sb = elem.encode("utf-8")
                    f.write(struct.pack("<Q", len(sb)))
                    f.write(sb)
        else:
            raise ValueError(f"Unsupported metadata value type: {vtype}")

    def _write_tensor_info(self, f, info: dict):
        """Write a single tensor info entry."""
        name_bytes = info["name"].encode("utf-8")
        f.write(struct.pack("<Q", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", info["n_dims"]))
        for d in info["dims"]:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<I", info["ggml_type"]))
        f.write(struct.pack("<Q", info["offset"]))

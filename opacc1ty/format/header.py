"""
BF2 binary file format specification.

The .bf2 format stores 2-bit quantized model weights with codebooks
and outlier channels, optimized for fast streaming into Metal GPU buffers.

File Layout:
┌──────────────────────────────────────┐
│ Magic: "BF2\0"             (4 bytes) │
│ Version: uint32            (4 bytes) │
│ Flags: uint32              (4 bytes) │
│ Model metadata JSON length (4 bytes) │
│ Model metadata JSON        (variable) │
│ Quantization config JSON   (4 + len) │
│ Layer count: uint32        (4 bytes) │
├──────────────────────────────────────┤
│ Layer 0 header             (64 bytes)│
│ Layer 0 codebooks           (varies) │
│ Layer 0 indices (packed)    (varies) │
│ Layer 0 outlier values      (varies) │
│ Layer 0 outlier indices     (varies) │
├──────────────────────────────────────┤
│ Layer 1 ...                          │
│ ...                                  │
├──────────────────────────────────────┤
│ Checksum: xxhash64          (8 bytes)│
└──────────────────────────────────────┘

Each layer header (64 bytes, packed struct):
  - name_len: uint16     (length of layer name string)
  - type: uint8          (0=quantized, 1=sparse_outlier, 2=fp16_passthrough)
  - out_features: uint32
  - in_features: uint32
  - n_groups: uint32     (number of codebook groups)
  - codebook_entries: uint8 (typically 4 for 2-bit)
  - sub_vector_size: uint8 (typically 8)
  - n_outliers: uint16   (number of outlier channels, 0 if none)
  - cb_offset: uint64    (byte offset to codebook data from layer start)
  - idx_offset: uint64   (byte offset to index data)
  - out_val_offset: uint64 (byte offset to outlier values, 0 if none)
  - out_idx_offset: uint64 (byte offset to outlier indices, 0 if none)
  - reserved: 14 bytes
"""

import struct
import json
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Tuple


MAGIC = b"BF2\x00"
VERSION = 1
HEADER_SIZE = 64  # Per-layer header, fixed size


class LayerType(IntEnum):
    QUANTIZED = 0
    SPARSE_OUTLIER = 1
    FP16_PASSTHROUGH = 2


@dataclass
class LayerFormat:
    """Describes the on-disk layout of one quantized layer."""
    name: str
    layer_type: LayerType
    out_features: int
    in_features: int
    n_groups: int = 0
    codebook_entries: int = 4
    sub_vector_size: int = 8
    n_outliers: int = 0
    cb_offset: int = 0
    idx_offset: int = 0
    out_val_offset: int = 0
    out_idx_offset: int = 0

    def pack(self) -> bytes:
        """Serialize to 64-byte binary header."""
        name_bytes = self.name.encode("utf-8")[:256]
        return struct.pack(
            "<H B I I I B B H Q Q Q Q 13s",
            len(name_bytes),
            self.layer_type,
            self.out_features,
            self.in_features,
            self.n_groups,
            self.codebook_entries,
            self.sub_vector_size,
            self.n_outliers,
            self.cb_offset,
            self.idx_offset,
            self.out_val_offset,
            self.out_idx_offset,
            b"\x00" * 13,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "LayerFormat":
        """Deserialize from 64-byte binary header."""
        (
            name_len, ltype, out_f, in_f, n_grp, cb_ent, svs, n_out,
            cb_off, idx_off, out_v_off, out_i_off, _
        ) = struct.unpack("<H B I I I B B H Q Q Q Q 13s", data)
        return cls(
            name="",  # name follows header
            layer_type=LayerType(ltype),
            out_features=out_f,
            in_features=in_f,
            n_groups=n_grp,
            codebook_entries=cb_ent,
            sub_vector_size=svs,
            n_outliers=n_out,
            cb_offset=cb_off,
            idx_offset=idx_off,
            out_val_offset=out_v_off,
            out_idx_offset=out_i_off,
        )


@dataclass
class BF2Header:
    """Top-level BF2 file header with model metadata."""
    version: int = VERSION
    flags: int = 0
    model_config: dict = field(default_factory=dict)
    quantize_config: dict = field(default_factory=dict)
    layer_count: int = 0

    def pack(self) -> bytes:
        """Serialize the file header."""
        model_json = json.dumps(self.model_config).encode("utf-8")
        quant_json = json.dumps(self.quantize_config).encode("utf-8")

        header = struct.pack(
            "<4s I I I",
            MAGIC,
            self.version,
            self.flags,
            len(model_json),
        )
        header += struct.pack("<I", len(model_json))
        header += model_json
        header += struct.pack("<I", len(quant_json))
        header += quant_json
        header += struct.pack("<I", self.layer_count)
        return header

    @classmethod
    def unpack(cls, data: bytes) -> Tuple["BF2Header", int]:
        """Deserialize the file header. Returns (header, bytes_consumed)."""
        magic, version, flags, model_len = struct.unpack("<4s I I I", data[:16])
        if magic != MAGIC:
            raise ValueError(f"Not a BF2 file: magic={magic!r}")

        pos = 16
        # model_json_len repeated (for alignment)
        model_json_len = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        model_json = json.loads(data[pos:pos+model_json_len].decode("utf-8"))
        pos += model_json_len

        quant_json_len = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        quant_json = json.loads(data[pos:pos+quant_json_len].decode("utf-8"))
        pos += quant_json_len

        layer_count = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4

        return cls(
            version=version,
            flags=flags,
            model_config=model_json,
            quantize_config=quant_json,
            layer_count=layer_count,
        ), pos

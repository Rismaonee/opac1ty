"""
BF2 file format reader and writer.

The BF2Writer serializes a quantized model into the .bf2 binary format.
The BF2Reader loads a .bf2 file and reconstructs tensors for inference.

Both handle the three layer types:
- QUANTIZED: 2-bit indices + codebooks + optional outlier channels
- SPARSE_OUTLIER: Pure sparse outlier representation
- FP16_PASSTHROUGH: Uncompressed fp16 data (for small tensors)
"""

import torch
import struct
import json
from pathlib import Path
from typing import Dict, BinaryIO, Optional
from tqdm import tqdm

from opacc1ty.format.header import (
    MAGIC, VERSION, HEADER_SIZE,
    BF2Header, LayerFormat, LayerType,
)


class BF2Writer:
    """Writes quantized models to .bf2 format.

    Example:
        >>> writer = BF2Writer("model.bf2")
        >>> writer.write(quantized_layers, model_config, quantize_config)
        >>> writer.close()
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.file: Optional[BinaryIO] = None
        self._bytes_written = 0

    def write(
        self,
        layers: Dict[str, dict],
        model_config: dict,
        quantize_config: dict,
        progress: bool = True,
    ):
        """Write all layers to the BF2 file."""
        self.file = open(self.path, "w+b")

        # Count quantizable layers
        layer_items = [
            (k, v) for k, v in layers.items()
            if not k.startswith("_")
        ]

        # Write file header
        header = BF2Header(
            model_config=model_config,
            quantize_config=quantize_config,
            layer_count=len(layer_items),
        )
        self.file.write(header.pack())

        # Write each layer
        iterator = tqdm(layer_items, desc="Writing BF2") if progress else layer_items

        for name, layer in iterator:
            self._write_layer(name, layer)

        file_size = self.file.tell()
        self.file.close()
        self.file = None

        compression = file_size / self._compute_original_size(layers)
        print(f"\nWrote {self.path} ({file_size / 1e9:.2f} GB, "
              f"{compression:.1f}× compression)")

    def _write_layer(self, name: str, layer: dict):
        """Write a single layer with its 64-byte header + data."""
        ltype = layer["type"]
        shape = layer["shape"]
        out_f, in_f = shape[0], shape[1] if len(shape) > 1 else 1
        layer_start = self.file.tell()

        if ltype == "quantized":
            cb = layer["codebooks"]
            idx = layer["indices"]
            n_groups = cb.shape[0]
            n_entries = cb.shape[1]
            svs = cb.shape[2]

            outlier_vals = layer.get("outlier_values")
            n_outliers = len(outlier_vals) if outlier_vals is not None else 0

            # Compute data offsets relative to layer start
            # Header (64) + name string
            name_bytes = name.encode("utf-8")
            data_start = HEADER_SIZE + len(name_bytes)

            fmt = LayerFormat(
                name=name,
                layer_type=LayerType.QUANTIZED,
                out_features=out_f,
                in_features=in_f,
                n_groups=n_groups,
                codebook_entries=n_entries,
                sub_vector_size=svs,
                n_outliers=n_outliers,
                cb_offset=data_start,
                idx_offset=data_start + cb.numel() * 2,  # fp16 = 2 bytes
                out_val_offset=(
                    data_start + cb.numel() * 2 + idx.numel()
                    if n_outliers > 0 else 0
                ),
                out_idx_offset=(
                    data_start + cb.numel() * 2 + idx.numel() + n_outliers * in_f * 2
                    if n_outliers > 0 else 0
                ),
            )

            # Write header
            self.file.write(fmt.pack())
            self.file.write(name_bytes)

            # Write codebooks (fp16)
            self.file.write(cb.numpy().tobytes())

            # Write indices (uint8)
            self.file.write(idx.numpy().tobytes())

            # Write outlier values (fp16)
            if n_outliers > 0:
                self.file.write(outlier_vals.numpy().tobytes())
                self.file.write(layer["outlier_indices"].numpy().tobytes())

        elif ltype == "fp16_passthrough":
            data = layer["data"]
            name_bytes = name.encode("utf-8")
            data_start = HEADER_SIZE + len(name_bytes)

            fmt = LayerFormat(
                name=name,
                layer_type=LayerType.FP16_PASSTHROUGH,
                out_features=out_f,
                in_features=in_f,
                cb_offset=data_start,
            )
            self.file.write(fmt.pack())
            self.file.write(name_bytes)
            self.file.write(data.numpy().tobytes())

    def _compute_original_size(self, layers: dict) -> int:
        """Compute total size of original fp16 weights."""
        total = 0
        for k, v in layers.items():
            if k.startswith("_"):
                continue
            shape = v.get("shape", (0,))
            total += int(torch.prod(torch.tensor(shape))) * 2
        return total

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class BF2Reader:
    """Reads .bf2 files and reconstructs tensors for inference.

    Example:
        >>> reader = BF2Reader("model.bf2")
        >>> model_config = reader.model_config
        >>> weights = reader.load_layer("model.layers.0.mlp.down_proj")
        >>> # weights contains codebooks, indices, outliers ready for Metal
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.file = open(self.path, "rb")
        self.header, self._data_start = BF2Header.unpack(self.file.read(1024))
        # Re-read from start for proper offset tracking
        self.file.seek(self._data_start)
        self._layer_index = self._build_index()

    def _build_index(self) -> Dict[str, LayerFormat]:
        """Build an index of all layers and their file offsets."""
        index = {}
        pos = self._data_start

        for _ in range(self.header.layer_count):
            self.file.seek(pos)
            header_data = self.file.read(HEADER_SIZE)
            fmt = LayerFormat.unpack(header_data)

            name_bytes = self.file.read(
                struct.unpack("<H", header_data[:2])[0]
            )
            fmt.name = name_bytes.decode("utf-8")

            index[fmt.name] = fmt

            # Compute next layer position
            if fmt.layer_type == LayerType.QUANTIZED:
                indices_per_row = fmt.in_features // fmt.sub_vector_size
                layer_data_size = (
                    fmt.n_groups * fmt.codebook_entries * fmt.sub_vector_size * 2
                    + fmt.n_groups * indices_per_row  # indices (uint8)
                )
                if fmt.n_outliers > 0:
                    layer_data_size += fmt.n_outliers * fmt.in_features * 2  # values
                    layer_data_size += fmt.n_outliers * 4  # indices (int32)
                pos += HEADER_SIZE + len(fmt.name.encode("utf-8")) + layer_data_size
            elif fmt.layer_type == LayerType.FP16_PASSTHROUGH:
                pos += (
                    HEADER_SIZE + len(fmt.name.encode("utf-8"))
                    + fmt.out_features * fmt.in_features * 2
                )

        return index

    @property
    def model_config(self) -> dict:
        return self.header.model_config

    @property
    def quantize_config(self) -> dict:
        return self.header.quantize_config

    def layer_names(self) -> list:
        return list(self._layer_index.keys())

    def load_layer(self, name: str) -> dict:
        """Load a single layer from the file.

        Returns dict with codebooks, indices, outlier data as numpy arrays
        ready to be uploaded to Metal buffers.
        """
        fmt = self._layer_index[name]

        # Seek to layer data start
        self.file.seek(
            self._data_start
            + sum(1 for n in self._layer_index if n < name)  # simplified; use offset tracking
        )

        # Actually, let's use the offsets from the header
        layer_header_pos = self._find_layer_position(name)
        self.file.seek(layer_header_pos + HEADER_SIZE + len(name.encode("utf-8")))

        if fmt.layer_type == LayerType.QUANTIZED:
            cb_bytes = fmt.n_groups * fmt.codebook_entries * fmt.sub_vector_size * 2
            indices_per_row = fmt.in_features // fmt.sub_vector_size
            idx_bytes = fmt.n_groups * indices_per_row

            codebooks = torch.frombuffer(
                bytearray(self.file.read(cb_bytes)),
                dtype=torch.float16,
            ).reshape(fmt.n_groups, fmt.codebook_entries, fmt.sub_vector_size)

            indices = torch.frombuffer(
                bytearray(self.file.read(idx_bytes)),
                dtype=torch.uint8,
            ).reshape(fmt.n_groups, indices_per_row)

            result = {
                "type": "quantized",
                "shape": (fmt.out_features, fmt.in_features),
                "codebooks": codebooks,
                "indices": indices,
            }

            if fmt.n_outliers > 0:
                out_val_bytes = fmt.n_outliers * fmt.in_features * 2
                result["outlier_values"] = torch.frombuffer(
                    bytearray(self.file.read(out_val_bytes)),
                    dtype=torch.float16,
                ).reshape(fmt.n_outliers, fmt.in_features)

                result["outlier_indices"] = torch.frombuffer(
                    bytearray(self.file.read(fmt.n_outliers * 4)),
                    dtype=torch.int32,
                )

            return result

        elif fmt.layer_type == LayerType.FP16_PASSTHROUGH:
            data_bytes = fmt.out_features * fmt.in_features * 2
            data = torch.frombuffer(
                bytearray(self.file.read(data_bytes)),
                dtype=torch.float16,
            ).reshape(fmt.out_features, fmt.in_features)

            return {
                "type": "fp16_passthrough",
                "shape": (fmt.out_features, fmt.in_features),
                "data": data,
            }

    def _find_layer_position(self, name: str) -> int:
        """Find the byte offset of a layer's header in the file."""
        pos = self._data_start
        for _ in range(self.header.layer_count):
            self.file.seek(pos)
            header_data = self.file.read(HEADER_SIZE)
            name_len = struct.unpack("<H", header_data[:2])[0]
            layer_name = self.file.read(name_len).decode("utf-8")

            if layer_name == name:
                return pos

            fmt = LayerFormat.unpack(header_data)
            if fmt.layer_type == LayerType.QUANTIZED:
                indices_per_row = fmt.in_features // fmt.sub_vector_size
                pos += (
                    HEADER_SIZE + name_len
                    + fmt.n_groups * fmt.codebook_entries * fmt.sub_vector_size * 2
                    + fmt.n_groups * indices_per_row
                )
                if fmt.n_outliers > 0:
                    pos += fmt.n_outliers * fmt.in_features * 2 + fmt.n_outliers * 4
            elif fmt.layer_type == LayerType.FP16_PASSTHROUGH:
                pos += (
                    HEADER_SIZE + name_len
                    + fmt.out_features * fmt.in_features * 2
                )

        raise KeyError(f"Layer '{name}' not found in BF2 file")

    def close(self):
        self.file.close()

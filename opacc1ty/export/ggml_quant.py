"""
GGML K-Quant implementation — Q2_K, Q3_K, Q4_K block quantization.

Matches llama.cpp's block_q2_K, block_q3_K, block_q4_K struct layouts exactly
so that GGUF files produced here load correctly in llama.cpp without patches.

Block formats (all use QK_K = 256 super-blocks):

Q2_K (2.625 bpw):
    2× fp16 (d, dmin) + 16× uint8 scales[4-bit scale + 4-bit min]
    + 64× uint8 quants[4× 2-bit per byte]
    → 84 bytes per 256 weights

Q3_K (3.4375 bpw):
    1× fp16 (d) + 32× uint8 hmask[high bit of each 3-bit quant]
    + 64× uint8 quants[4× 2-bit low bits per byte]
    + 12× uint8 scales[16× 6-bit packed]
    → 110 bytes per 256 weights

Q4_K (4.5 bpw):
    2× fp16 (d, dmin) + 12× uint8 scales[8× 6-bit scale + 8× 6-bit min]
    + 128× uint8 quants[2× 4-bit per byte]
    → 144 bytes per 256 weights
"""

import struct
import torch
import numpy as np
from typing import Tuple

QK_K = 256
K_SCALE_SIZE = 12


def quantize_q2_k(x: torch.Tensor) -> bytes:
    """Quantize flat float32 tensor to Q2_K block format.

    16 sub-blocks of 16 elements each.
    Per sub-block: 4-bit scale (shared with sub-block min).
    Per element: 2-bit quant (0-3).
    Super-block: d (scale) + dmin (min offset).
    Quant formula: x_hat = d * q + dmin

    Returns bytes suitable for writing directly to GGUF tensor data section.
    """
    x = x.float()
    n = x.numel()
    assert n % QK_K == 0, f"tensor size {n} not divisible by {QK_K}"

    n_blocks = n // QK_K
    out = bytearray()

    for b in range(n_blocks):
        block = x[b * QK_K : (b + 1) * QK_K]  # 256 elements

        # Find super-block range
        x_min = block.min().item()
        x_max = block.max().item()

        # d: super-block scale for sub-block scales and mins
        # dmin: super-block minimum offset
        dmin = float(np.float16(x_min))
        d = float(np.float16((x_max - x_min) / 3.0)) if x_max > x_min else float(np.float16(1.0 / 256.0))

        # 16 sub-blocks of 16 elements each
        scales = bytearray(16)
        quants = bytearray(64)

        for sb in range(16):
            sub = block[sb * 16 : (sb + 1) * 16]
            sub_min = sub.min().item()
            sub_max = sub.max().item()

            # 4-bit scale and 4-bit min for this sub-block
            # Each is quantized to 4 bits (0-15)
            sub_range = (sub_max - sub_min) if sub_max > sub_min else 1.0
            scale_4bit = int(np.clip(sub_range / (3.0 * max(d, 1e-8)), 0, 15))
            min_4bit = int(np.clip((sub_min - dmin) / max(d, 1e-8), 0, 15))
            scales[sb] = (min_4bit & 0xF) | ((scale_4bit & 0xF) << 4)

            # 2-bit quantization of each element
            for j in range(16):
                val = sub[j].item()
                # Reconstruct sub-min and sub-scale from packed values
                smin = dmin + min_4bit * d
                sscale = scale_4bit * d
                q = int(np.clip(round((val - smin) / max(sscale, 1e-8)), 0, 3))
                byte_idx = sb * 16 + j
                quants[byte_idx // 4] |= (q & 0x3) << ((byte_idx % 4) * 2)

        # Write super-block header
        out += struct.pack('<e', d)      # ggml_half d
        out += struct.pack('<e', dmin)   # ggml_half dmin
        out += bytes(scales)             # uint8_t scales[16]
        out += bytes(quants)             # uint8_t qs[64]

    return bytes(out)


def quantize_q3_k(x: torch.Tensor) -> bytes:
    """Quantize flat float32 tensor to Q3_K block format.

    16 sub-blocks of 16 elements each.
    Per element: 3-bit quant (0-7).
    High bit stored in hmask[32], low 2 bits in qs[64].
    Per sub-block: 6-bit scale packed in scales[12].
    Super-block: d (scale).

    Quant formula: x_hat = d * q * scale[sub_block]
    """
    x = x.float()
    n = x.numel()
    assert n % QK_K == 0

    n_blocks = n // QK_K
    out = bytearray()

    for b in range(n_blocks):
        block = x[b * QK_K : (b + 1) * QK_K]

        x_max = block.abs().max().item()
        d = float(np.float16(x_max / 7.0)) if x_max > 0 else float(np.float16(1.0 / 256.0))

        hmask = bytearray(32)   # high bits
        qs = bytearray(64)      # low 2 bits
        scales_6bit = [0] * 16  # 6-bit values for each sub-block

        for sb in range(16):
            sub = block[sb * 16 : (sb + 1) * 16]
            sub_max = sub.abs().max().item()
            # 6-bit scale (0-63)
            sc = int(np.clip(sub_max / max(d, 1e-8), 0, 63))
            scales_6bit[sb] = sc

            for j in range(16):
                val = sub[j].item()
                q = int(np.clip(round(val / max(d * max(sc, 1), 1e-8)), -4, 3))
                qu = q + 4  # shift to 0-7 range (3-bit unsigned)

                # High bit into hmask
                byte_idx = (sb * 16 + j)
                if qu & 0x4:
                    hmask[byte_idx // 8] |= (1 << (byte_idx % 8))
                # Low 2 bits into qs
                qs[byte_idx // 4] |= (qu & 0x3) << ((byte_idx % 4) * 2)

        # Pack 16 × 6-bit scales into 12 bytes
        scales_packed = bytearray(K_SCALE_SIZE)
        for i in range(16):
            bits = scales_6bit[i] & 0x3F
            bit_pos = i * 6
            byte_pos = bit_pos // 8
            bit_off = bit_pos % 8
            scales_packed[byte_pos] |= (bits << bit_off) & 0xFF
            if byte_pos + 1 < K_SCALE_SIZE:
                scales_packed[byte_pos + 1] |= (bits >> (8 - bit_off)) & 0xFF

        out += struct.pack('<e', d)
        out += bytes(hmask)
        out += bytes(qs)
        out += bytes(scales_packed)

    return bytes(out)


def quantize_q4_k(x: torch.Tensor) -> bytes:
    """Quantize flat float32 tensor to Q4_K block format.

    8 sub-blocks of 32 elements each (different from Q2/Q3!).
    Per element: 4-bit quant (0-15).
    Per sub-block: 6-bit scale + 6-bit min packed in scales[12].
    Super-block: d (scale) + dmin (min offset).

    Quant formula: x_hat = d * q * scale[sb] + dmin * min[sb]
    """
    x = x.float()
    n = x.numel()
    assert n % QK_K == 0

    n_blocks = n // QK_K
    out = bytearray()

    for b in range(n_blocks):
        block = x[b * QK_K : (b + 1) * QK_K]

        x_min = block.min().item()
        x_max = block.max().item()
        d = float(np.float16((x_max - x_min) / 15.0)) if x_max > x_min else float(np.float16(1.0 / 256.0))
        dmin = float(np.float16(x_min))

        quants = bytearray(QK_K // 2)  # 128 bytes, 2× 4-bit per byte
        scales_6bit = []  # 8 scale values + 8 min values
        mins_6bit = []

        for sb in range(8):
            sub = block[sb * 32 : (sb + 1) * 32]
            sub_min = sub.min().item()
            sub_max = sub.max().item()
            sub_range = sub_max - sub_min if sub_max > sub_min else 1.0

            sc = int(np.clip(sub_range / (15.0 * max(d, 1e-8)), 0, 63))
            mn = int(np.clip((sub_min - dmin) / max(d, 1e-8), 0, 63))
            scales_6bit.append(sc)
            mins_6bit.append(mn)

            for j in range(32):
                val = sub[j].item()
                q = int(np.clip(round((val - sub_min) / max(sub_range / 15.0, 1e-8)), 0, 15))
                byte_idx = sb * 32 + j
                if byte_idx % 2 == 0:
                    quants[byte_idx // 2] = q & 0xF
                else:
                    quants[byte_idx // 2] |= (q & 0xF) << 4

        # Pack 8 × 6-bit scales + 8 × 6-bit mins into 12 bytes
        all_6bit = scales_6bit + mins_6bit  # 16 values, 6 bits each
        scales_packed = bytearray(K_SCALE_SIZE)
        for i in range(16):
            bits = all_6bit[i] & 0x3F
            bit_pos = i * 6
            byte_pos = bit_pos // 8
            bit_off = bit_pos % 8
            scales_packed[byte_pos] |= (bits << bit_off) & 0xFF
            if byte_pos + 1 < K_SCALE_SIZE:
                scales_packed[byte_pos + 1] |= (bits >> (8 - bit_off)) & 0xFF

        out += struct.pack('<e', d)
        out += struct.pack('<e', dmin)
        out += bytes(scales_packed)
        out += bytes(quants)

    return bytes(out)


QUANT_FORMATS = {
    "Q2_K": (quantize_q2_k, 10),  # GGML_TYPE_Q2_K = 10
    "Q3_K": (quantize_q3_k, 11),  # GGML_TYPE_Q3_K = 11
    "Q4_K": (quantize_q4_k, 12),  # GGML_TYPE_Q4_K = 12
}

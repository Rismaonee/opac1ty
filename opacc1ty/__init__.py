"""
Opacc1ty — 2-bit quantization with fused Metal dequant kernels for Apple Silicon.

Provides a complete pipeline for compressing LLM weights to 2 bits/weight
while maintaining inference speed through custom Metal GPU kernels that
fuse dequantization directly into the matmul inner loop.

Usage:
    >>> import opacc1ty as bf
    >>> bf.quantize("model.safetensors", bits=2, output="model.bf2")
    >>> engine = bf.Engine("model.bf2")
    >>> engine.generate("Hello, world!")
"""

from opacc1ty.quantize.vq import VectorQuantizer
from opacc1ty.quantize.codebook import CodebookLearner
from opacc1ty.quantize.outlier import OutlierDetector
from opacc1ty.format.bf2 import BF2Writer, BF2Reader
from opacc1ty.format.header import BF2Header, LayerFormat

__version__ = "0.1.0"
__all__ = [
    "VectorQuantizer",
    "CodebookLearner",
    "OutlierDetector",
    "BF2Writer",
    "BF2Reader",
    "BF2Header",
    "LayerFormat",
]

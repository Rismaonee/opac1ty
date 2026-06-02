"""
Metal kernel management utilities for Opacc1ty.

Handles:
- Compiling Metal shader source to Metallib at import time
- Detecting Apple Silicon GPU capabilities
- Managing Metal device, command queue, and buffer allocation
- Dispatching fused dequant+matmul kernels with correct parameters
"""

import os
import subprocess
import tempfile
import ctypes
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


# Metal kernels directory
KERNELS_DIR = Path(__file__).parent.parent.parent / "kernels"


@dataclass
class GPUInfo:
    """Apple Silicon GPU capabilities."""
    name: str
    max_threads_per_threadgroup: int
    max_total_threads_per_threadgroup: int
    threadgroup_memory_size: int
    max_buffer_size: int
    supports_family_apple8: bool  # M3 or newer
    supports_family_apple7: bool  # M2
    supports_family_apple6: bool  # M1
    unified_memory_gb: float
    gpu_cores: int


def get_metal_device_info() -> Optional[GPUInfo]:
    """Query the default Metal device for capabilities.

    Returns None if no Apple Silicon GPU is available.
    """
    try:
        import objc
        from Foundation import NSBundle

        # Load Metal framework
        bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/Metal.framework")
        if not bundle.load():
            return None

        MetalDevice = objc.lookUpClass("MTLCreateSystemDefaultDevice")
        if MetalDevice is None:
            return None

        device = MetalDevice()
        if device is None:
            return None

        return GPUInfo(
            name=str(device.name()),
            max_threads_per_threadgroup=int(device.maxThreadsPerThreadgroup()),
            max_total_threads_per_threadgroup=int(
                device.maxTotalThreadsPerThreadgroup()
            ),
            threadgroup_memory_size=int(device.maxThreadgroupMemoryLength()),
            max_buffer_size=int(device.maxBufferLength()),
            supports_family_apple8=device.supportsFamily_(1008),  # MTLGPUFamilyApple8
            supports_family_apple7=device.supportsFamily_(1007),
            supports_family_apple6=device.supportsFamily_(1006),
            unified_memory_gb=device.recommendedMaxWorkingSetSize() / 1e9 if hasattr(device, 'recommendedMaxWorkingSetSize') else 0,
            gpu_cores=0,  # Not directly exposed by Metal API
        )
    except ImportError:
        return None


class MetalKernelManager:
    """Manages Metal compute pipeline for Opacc1ty inference.

    Compiles .metal shader files into a Metallib, manages Metal device
    and command queue, and provides methods to dispatch the fused
    dequant+GEMM/GEMV kernels.

    Example:
        >>> mgr = MetalKernelManager()
        >>> mgr.compile_kernels()
        >>> mgr.dequant_gemv(codebooks_buf, indices_buf, input_buf,
        ...                   output_buf, M=4096, K=4096)
    """

    def __init__(self, device=None):
        self.metallib_path: Optional[Path] = None
        self._device = device
        self._command_queue = None
        self._library = None
        self._pipelines = {}

    def compile_kernels(self) -> Path:
        """Compile all .metal kernel files into a single .metallib.

        Uses the system Metal compiler (xcrun metal) to produce
        GPU binaries optimized for the current device.

        Returns:
            Path to the compiled .metallib file.
        """
        metal_files = list(KERNELS_DIR.glob("*.metal"))
        if not metal_files:
            raise FileNotFoundError(
                f"No .metal files found in {KERNELS_DIR}"
            )

        with tempfile.NamedTemporaryFile(
            suffix=".metallib", delete=False
        ) as tmp:
            metallib_path = Path(tmp.name)

        # Compile each .metal file to .air (Apple IR), then link
        air_files = []
        for mf in metal_files:
            air_path = mf.with_suffix(".air")
            cmd = [
                "xcrun", "-sdk", "macosx", "metal",
                "-c", str(mf),
                "-o", str(air_path),
                "-arch", "air64",
                "-ffast-math",
                "-fno-fast-math",  # careful with precision
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            air_files.append(air_path)

        # Link .air files into .metallib
        link_cmd = [
            "xcrun", "-sdk", "macosx", "metallib",
            *[str(a) for a in air_files],
            "-o", str(metallib_path),
        ]
        subprocess.run(link_cmd, check=True, capture_output=True)

        # Clean up .air files
        for a in air_files:
            a.unlink()

        self.metallib_path = metallib_path
        return metallib_path

    def load_library(self):
        """Load the compiled Metallib into the Metal device.

        Call this after compile_kernels() and before dispatching kernels.
        """
        if self.metallib_path is None:
            self.compile_kernels()

        # Use ctypes to call Metal C API
        # This is the C-level path; for production use, prefer the
        # Objective-C runtime via PyObjC or a native C extension
        try:
            import objc
            from Metal import (
                MTLCreateSystemDefaultDevice,
                MTLLibrary,
                MTLComputePipelineState,
            )

            self._device = MTLCreateSystemDefaultDevice()
            if self._device is None:
                raise RuntimeError("No Metal device found")

            self._command_queue = self._device.newCommandQueue()

            # Load library
            options = 0  # MTLCompileOptions
            self._library = self._device.newLibraryWithURL_error_(
                objc.pathForURL(self.metallib_path),
                options,
                None,
            )
            if self._library is None:
                raise RuntimeError(
                    f"Failed to load metallib: {self.metallib_path}"
                )

            # Pre-load all compute pipelines
            for kernel_name in ["dequant_gemv_2bit", "dequant_gemv_2bit_tiled",
                                "dequant_gemm_2bit"]:
                func = self._library.newFunctionWithName_(kernel_name)
                if func:
                    self._pipelines[kernel_name] = (
                        self._device.newComputePipelineStateWithFunction_error_(
                            func, None
                        )[0]
                    )

        except ImportError:
            # Fallback: use ctypes directly
            self._load_via_ctypes()

    def dequant_gemv(
        self,
        codebooks_buf,   # Metal buffer
        indices_buf,     # Metal buffer
        input_buf,       # Metal buffer
        output_buf,      # Metal buffer
        M: int,
        K: int,
        outlier_vals_buf=None,
        outlier_idx_buf=None,
        n_outliers: int = 0,
    ):
        """Dispatch the fused dequant+GEMV kernel.

        Args:
            codebooks_buf: Metal buffer with fp16 codebook values.
            indices_buf: Metal buffer with packed 2-bit indices.
            input_buf: Metal buffer with fp16 input vector (K elements).
            output_buf: Metal buffer for fp16 output vector (M elements).
            M: Output dimension (rows of weight matrix).
            K: Input dimension (columns of weight matrix).
            outlier_vals_buf: Optional sparse outlier values.
            outlier_idx_buf: Optional sparse outlier indices.
            n_outliers: Number of outlier channels.
        """
        import objc
        from Metal import MTLSize

        K_groups = K // 8

        # Choose kernel variant
        threads_needed = M
        max_threads = self._device.maxThreadsPerThreadgroup()

        if threads_needed <= max_threads:
            kernel = self._pipelines["dequant_gemv_2bit"]
            threadgroup_size = min(256, max_threads)
            grid_size = MTLSize(M, 1, 1)
            group_size = MTLSize(threadgroup_size, 1, 1)
            rows_per_thread = 1
        else:
            kernel = self._pipelines["dequant_gemv_2bit_tiled"]
            rows_per_thread = (M + max_threads - 1) // max_threads
            threadgroup_size = min(256, max_threads)
            grid_size = MTLSize(
                (M + rows_per_thread - 1) // rows_per_thread, 1, 1
            )
            group_size = MTLSize(threadgroup_size, 1, 1)

        # Build command buffer
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        encoder.setComputePipelineState_(kernel)
        encoder.setBuffer_offset_atIndex_(codebooks_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(indices_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(input_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 3)

        # Encode scalar parameters via constant buffer
        import struct
        params = struct.pack("<III", M, K, K_groups)
        params_buf = self._device.newBufferWithBytes_length_options_(
            params, len(params), 0  # MTLResourceStorageModeShared
        )
        encoder.setBuffer_offset_atIndex_(params_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(params_buf, 4, 5)
        encoder.setBuffer_offset_atIndex_(params_buf, 8, 6)

        if outlier_vals_buf and outlier_idx_buf:
            encoder.setBuffer_offset_atIndex_(outlier_vals_buf, 0, 7)
            encoder.setBuffer_offset_atIndex_(outlier_idx_buf, 0, 8)
            encoder.setBytes_length_atIndex_(
                struct.pack("<I", n_outliers), 4, 9
            )

        if rows_per_thread > 1:
            encoder.setBytes_length_atIndex_(
                struct.pack("<I", rows_per_thread), 4, 7
            )

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            grid_size, group_size
        )
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

    def _load_via_ctypes(self):
        """Fallback Metal loading via ctypes (no PyObjC dependency)."""
        import ctypes
        import ctypes.util

        metal = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("Metal")
        )

        # Create default device
        device = ctypes.c_void_p()
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        self._device_ptr = metal.MTLCreateSystemDefaultDevice()

        # Create command queue
        # self._device.newCommandQueue() via objc_msgSend
        # This is complex in pure ctypes — for now, raise a helpful error
        raise RuntimeError(
            "Pure-ctypes Metal backend not yet implemented. "
            "Install PyObjC: pip install pyobjc-framework-Metal"
        )

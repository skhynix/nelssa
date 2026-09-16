#!/usr/bin/env python3
"""
Setup script for the NELSSA fused LSE merge CUDA extension.
Compiles merge_kernel.cu into vllm_merge_ext (in-place).
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# H200 = sm_90. Pin the arch so the kernel is pre-compiled (no JIT, low latency).
NVCC_FLAGS = ["-O3", "-arch=sm_90", "--use_fast_math"]

ext_modules = [
    CUDAExtension(
        name="vllm_merge_ext",
        sources=["merge_kernel.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": NVCC_FLAGS},
    )
]

setup(
    name="vllm_merge_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

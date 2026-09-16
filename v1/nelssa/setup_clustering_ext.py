#!/usr/bin/env python3
"""
Setup script for C++ Clustering Extension
Compiles the multi-threaded CPU reorganization kernel
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Get OpenMP flags
try:
    from setuptools import _distutils
    import subprocess
    # Check if OpenMP is available
    try:
        subprocess.check_output(['gcc', '-fopenmp', '-E', '-x', 'c', '-'], input=b'')
        extra_compile_args = ['-fopenmp', '-O3', '-march=native']
        extra_link_args = ['-fopenmp']
    except:
        extra_compile_args = ['-O3']
        extra_link_args = []
except:
    extra_compile_args = ['-O3']
    extra_link_args = []

ext_modules = [
    CppExtension(
        name="vllm_clustering_ext",
        sources=["clustering_cpu.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="vllm_clustering_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
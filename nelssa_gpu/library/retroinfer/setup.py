import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

src_dir = "retroinfer_kernels/src"
cutlass_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../cutlass")

ext_modules = [
    CppExtension(
        "retroinfer_kernels.WaveBuffer",
        sources=[f"{src_dir}/wave_buffer_cpu.cpp"],
        include_dirs=["/usr/local/cuda-12/include"],
        library_dirs=["/usr/local/lib"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
    CUDAExtension(
        "retroinfer_kernels.Copy",
        sources=[f"{src_dir}/gather_copy.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        },
        extra_link_args=["-lcuda", "-lcudart"],
    ),
    CUDAExtension(
        "retroinfer_kernels.gemm_softmax",
        sources=[f"{src_dir}/batch_gemm_softmax.cu"],
        include_dirs=[
            f"{cutlass_dir}/include",
            f"{cutlass_dir}/examples/common",
            f"{cutlass_dir}/tools/util/include",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        },
        extra_link_args=["-lcuda", "-lcudart"],
    ),
]


setup(
    name="retroinfer_kernels",
    version="0.1",
    packages=["retroinfer_kernels"],
    description="RetroInfer kernels and modules",
    long_description="A collection of CUDA and C++ extensions for RetroInfer.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["pybind11", "torch"],
    python_requires=">=3.10",
)

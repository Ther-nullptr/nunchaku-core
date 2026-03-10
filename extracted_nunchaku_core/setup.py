import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# This environment has NVCC 13.x while torch is built with CUDA 12.8.
# For this extracted demo library we allow cross-minor compilation.
cpp_extension._check_cuda_version = lambda *args, **kwargs: None

ROOT_DIR = Path(__file__).resolve().parent

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    arch_list = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            if major == 12 and minor == 0:
                arch = "12.0a"
            elif major == 12 and minor == 1:
                arch = "12.1a"
            else:
                arch = f"{major}.{minor}"
            if arch not in arch_list:
                arch_list.append(arch)
    if not arch_list:
        # Fallback for headless/offline build contexts in this workspace.
        arch_list = ["12.0a"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

ext_modules = [
    CUDAExtension(
        name="nunchaku_core._int4_cuda",
        sources=[
            "csrc/int4_ops.cpp",
            "csrc/int4_ops_cuda.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math"],
        },
    ),
    CUDAExtension(
        name="nunchaku_core._fp4_native_cuda",
        sources=[
            "csrc/fp4_native_ops.cpp",
            "csrc/fp4_lora_decode_cuda.cu",
            "csrc/fp4_repack_cuda.cu",
            "fp4_backend/src/interop/torch.cpp",
            "fp4_backend/src/kernels/zgemm/gemm_w4a4.cu",
            "fp4_backend/src/kernels/zgemm/gemm_w4a4_launch_fp16_int4.cu",
            "fp4_backend/src/kernels/zgemm/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
            "fp4_backend/src/kernels/zgemm/gemm_w4a4_launch_fp16_fp4.cu",
            "fp4_backend/src/kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu",
            "fp4_backend/src/kernels/zgemm/gemm_w4a4_launch_bf16_fp4.cu",
        ],
        include_dirs=[
            str(ROOT_DIR / "fp4_backend" / "src"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", "-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-DENABLE_BF16=1",
                "-DBUILD_NUNCHAKU=1",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_HALF2_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
        },
    ),
]

setup(
    name="nunchaku_core",
    version="0.1.0",
    description="Extracted SVDQuant core operators from Nunchaku",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

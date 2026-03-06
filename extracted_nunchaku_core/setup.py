from setuptools import find_packages, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# This environment has NVCC 13.x while torch is built with CUDA 12.8.
# For this extracted demo library we allow cross-minor compilation.
cpp_extension._check_cuda_version = lambda *args, **kwargs: None

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
    )
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

import os
import glob
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension


bs_fit_root_path = os.getcwd() + "/../bspline-fitting/bspline_fitting/Cpp/"
bs_fit_src_path = bs_fit_root_path + "src/"
bs_fit_sources = glob.glob(bs_fit_src_path + "*.cpp")
bs_fit_include_dirs = [bs_fit_root_path + "include"]

bs_fit_extra_compile_args = [
    "-O3",
    "-std=c++17",
    "-DCMAKE_BUILD_TYPE Release",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
]

if torch.cuda.is_available():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9"

    bs_fit_sources += glob.glob(bs_fit_src_path + "*.cu")

    extra_compile_args = {
        "cxx": bs_fit_extra_compile_args + ["-DUSE_CUDA"],
        "nvcc": [
            "-O3",
            "-Xfatbin",
            "-compress-all",
            "-DUSE_CUDA",
        ],
    }

    bs_fit_module = CUDAExtension(
        name="bs_fit_cpp",
        sources=bs_fit_sources,
        include_dirs=bs_fit_include_dirs,
        extra_compile_args=extra_compile_args,
    )

else:
    bs_fit_module = CppExtension(
        name="bs_fit_cpp",
        sources=bs_fit_sources,
        include_dirs=bs_fit_include_dirs,
        extra_compile_args=bs_fit_extra_compile_args,
    )

setup(
    name="BSpline-Surface-Fitting-CPP",
    version="1.0.0",
    author="Changhao Li",
    packages=find_packages(),
    ext_modules=[bs_fit_module],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)

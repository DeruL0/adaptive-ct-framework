from __future__ import annotations

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cxx_flags = []
if os.name == "nt":
    cxx_flags.extend(["/wd4624", "/wd4819"])


setup(
    name="adaptive_ct",
    version="0.1.0",
    packages=find_packages() + ["adaptive_ct_native"],
    package_dir={"adaptive_ct_native": "native/adaptive_ct_native"},
    ext_modules=[
        CUDAExtension(
            name="adaptive_ct_native._C",
            sources=[
                "native/ext.cpp",
                "native/projector.cu",
                "native/dynamic_voxel.cu",
                "native/bernstein_octree.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

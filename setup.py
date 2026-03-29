from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="simd",
        sources=["simd.pyx"],
        # sources=["simdMAC.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2"],
        # extra_compile_args=["-O2", "-arch", "arm64"],
        # extra_link_args=["-arch", "arm64"]
    )
]

setup(
    ext_modules = cythonize(extensions)
)
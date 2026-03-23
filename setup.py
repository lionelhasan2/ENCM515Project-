from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="simd",
        sources=["simd.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2"],
    )
]

setup(
    ext_modules = cythonize(extensions)
)
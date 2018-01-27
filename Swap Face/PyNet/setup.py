try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("upsample_function",
                             sources=["upsample_cython.pyx", "upsample_kernel.c"],
                             include_dirs=[numpy.get_include()])],
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("im2col",
                             sources=["im2col_cython.pyx"],
                             include_dirs=[numpy.get_include()])],
)
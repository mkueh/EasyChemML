from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import sys

#!python
#cython: language_level=3

if sys.platform.startswith("win"):
     # compile args from
     # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
     compile_args = ['/O2', '/openmp']
     link_args = []
else:
     compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-O3', '-ffast-math']
     link_args = []
     compile_args.append("-fopenmp")
     link_args.append("-fopenmp")
     compile_args.append("-std=c++11")
     link_args.append("-std=c++11")

setup(ext_modules = cythonize(
          Extension(
               'CSMART_IsColumneRelevant',
               ["CSMART_IsColumneRelevant.pyx"],
               extra_compile_args=compile_args, extra_link_args=link_args)))

"""
setup(ext_modules = cythonize(
          Extension(
               'SFF_Generator',
               ["SFF_Generator.py"],
               extra_compile_args=compile_args, extra_link_args=link_args)))
"""


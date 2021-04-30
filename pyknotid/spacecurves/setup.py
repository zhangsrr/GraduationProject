from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_modules = [
        Extension("chelpers", ["chelpers.pyx"], libraries=["m"]),
        ]

setup(
  name = 'chelpers',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [numpy.get_include()],
  ext_modules = ext_modules
)

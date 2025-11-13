from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

def get_numpy_include():
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        # During build, we can use the system numpy include path
        return os.path.join(sys.prefix, 'lib', 'python' + sys.version[:3], 'site-packages', 'numpy', 'core', 'include')

# Define the Cython extension
extensions = [
    Extension(
        "dso.cyfunc",
        ["dso/cyfunc.pyx"],
        include_dirs=[get_numpy_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="dso",
    version="1.0dev",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19,<1.24",  # More flexible version range
        "tensorflow==1.14",
        "numba==0.53.1",
        "sympy",
        "pandas",
        "scikit-learn",
        "click",
        "deap",
        "pathos",
        "seaborn",
        "tqdm",
        "pyyaml",
        "prettytable",
        "pytest",
        "cython",
        "commentjson",
        "progress"
    ],
    ext_modules=extensions,
    python_requires=">=3.7",
)

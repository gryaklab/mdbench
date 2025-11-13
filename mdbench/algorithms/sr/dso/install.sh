#!/bin/bash
set -e

# Clean pip cache
python -m pip cache purge

# Clean conda cache
conda clean -a -y

# Ensure we're in the right environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate md-bench-dso

# Verify Python version
python --version | grep "Python 3.7" || (echo "Error: Python 3.7 is required" && exit 1)

# Ensure numpy is installed with the correct version for Python 3.7
conda install -y numpy=1.19

# Build the package
python setup.py build_ext --inplace

# Install the DSO package using PEP 517
python -m pip install --use-pep517 --no-deps -e .
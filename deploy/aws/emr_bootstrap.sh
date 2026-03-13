#!/bin/bash
# Bootstrap script for EMR GPU nodes: install Python dependencies for Hugging Face + PyTorch.
# Run on both master and core/task nodes.
set -e
sudo pip3 install --upgrade pip
sudo pip3 install 'pyspark>=3.5.0' 'torch>=2.0.0' 'transformers>=4.35.0' 'pandas>=2.0.0' 'pyarrow>=14.0.0'

#!/usr/bin/env bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 32" MAX_JOBS=128 # parallel compiling (Optional)
python3 setup.py install  # or pip install -e .

#!/bin/bash
# Soft Sensor Benchmark Experiments
# 4 models x 2 configurations = 8 experiments

# DLinear
python experiments/TSFsoftsensor_test/dlinear_excl_target.py
python experiments/TSFsoftsensor_test/dlinear_incl_target.py

# PatchTST
python experiments/TSFsoftsensor_test/patchtst_excl_target.py
python experiments/TSFsoftsensor_test/patchtst_incl_target.py

# iTransformer
python experiments/TSFsoftsensor_test/itransformer_excl_target.py
python experiments/TSFsoftsensor_test/itransformer_incl_target.py

# TimeXer
python experiments/TSFsoftsensor_test/timexer_excl_target.py
python experiments/TSFsoftsensor_test/timexer_incl_target.py

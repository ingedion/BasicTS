#!/bin/bash
# EthyDistillation Soft Sensor Benchmark Experiments
# 4 models x 2 configurations = 8 experiments

# DLinear
python experiments/EthyDistillation_benchmark/dlinear_excl_target.py
python experiments/EthyDistillation_benchmark/dlinear_incl_target.py

# PatchTST
python experiments/EthyDistillation_benchmark/patchtst_excl_target.py
python experiments/EthyDistillation_benchmark/patchtst_incl_target.py

# iTransformer
python experiments/EthyDistillation_benchmark/itransformer_excl_target.py
python experiments/EthyDistillation_benchmark/itransformer_incl_target.py

# TimeXer
python experiments/EthyDistillation_benchmark/timexer_excl_target.py
python experiments/EthyDistillation_benchmark/timexer_incl_target.py

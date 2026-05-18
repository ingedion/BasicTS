#!/bin/bash
# Debutanizer Soft Sensor Benchmark Experiments
# 4 models x 2 configurations = 8 experiments

# DLinear
python experiments/Debutanizer_benchmark/dlinear_excl_target.py
python experiments/Debutanizer_benchmark/dlinear_incl_target.py

# PatchTST
python experiments/Debutanizer_benchmark/patchtst_excl_target.py
python experiments/Debutanizer_benchmark/patchtst_incl_target.py

# iTransformer
python experiments/Debutanizer_benchmark/itransformer_excl_target.py
python experiments/Debutanizer_benchmark/itransformer_incl_target.py

# TimeXer
python experiments/Debutanizer_benchmark/timexer_excl_target.py
python experiments/Debutanizer_benchmark/timexer_incl_target.py

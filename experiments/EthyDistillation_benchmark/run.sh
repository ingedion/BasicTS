#!/bin/bash
# EthyDistillation Soft Sensor Benchmark Experiments
# Traditional baselines + Deep learning models

# ============ Traditional Baselines (NEW) ============

# NLinear (univariate autoregressive)
python experiments/EthyDistillation_benchmark/nlinear_univariate.py

# PLS (linear multivariate, non-gradient)
python experiments/EthyDistillation_benchmark/pls.py

# SVR (nonlinear static, non-gradient) — may be slow due to high dimensionality
python experiments/EthyDistillation_benchmark/svr.py

# LSTM (nonlinear temporal)
python experiments/EthyDistillation_benchmark/lstm_excl_target.py
python experiments/EthyDistillation_benchmark/lstm_incl_target.py

# ============ Deep Learning Models (already trained, uncomment to re-run) ============

# # DLinear
# python experiments/EthyDistillation_benchmark/dlinear_excl_target.py
# python experiments/EthyDistillation_benchmark/dlinear_incl_target.py

# # PatchTST
# python experiments/EthyDistillation_benchmark/patchtst_excl_target.py
# python experiments/EthyDistillation_benchmark/patchtst_incl_target.py

# # iTransformer
# python experiments/EthyDistillation_benchmark/itransformer_excl_target.py
# python experiments/EthyDistillation_benchmark/itransformer_incl_target.py

# # TimeXer
# python experiments/EthyDistillation_benchmark/timexer_excl_target.py
# python experiments/EthyDistillation_benchmark/timexer_incl_target.py

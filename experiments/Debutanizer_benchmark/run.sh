#!/bin/bash
# Debutanizer Soft Sensor Benchmark Experiments
# Traditional baselines + Deep learning models

# ============ Traditional Baselines (NEW) ============

# NLinear (univariate autoregressive)
python experiments/Debutanizer_benchmark/nlinear_univariate.py

# PLS (linear multivariate, non-gradient)
python experiments/Debutanizer_benchmark/pls.py

# SVR (nonlinear static, non-gradient)
python experiments/Debutanizer_benchmark/svr.py

# LSTM (nonlinear temporal)
python experiments/Debutanizer_benchmark/lstm_excl_target.py
python experiments/Debutanizer_benchmark/lstm_incl_target.py

# ============ Deep Learning Models (already trained, uncomment to re-run) ============

# # DLinear
# python experiments/Debutanizer_benchmark/dlinear_excl_target.py
# python experiments/Debutanizer_benchmark/dlinear_incl_target.py

# # PatchTST
# python experiments/Debutanizer_benchmark/patchtst_excl_target.py
# python experiments/Debutanizer_benchmark/patchtst_incl_target.py

# # iTransformer
# python experiments/Debutanizer_benchmark/itransformer_excl_target.py
# python experiments/Debutanizer_benchmark/itransformer_incl_target.py

# # TimeXer
# python experiments/Debutanizer_benchmark/timexer_excl_target.py
# python experiments/Debutanizer_benchmark/timexer_incl_target.py

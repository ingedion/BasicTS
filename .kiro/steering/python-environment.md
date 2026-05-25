# Python Environment

When executing any Python-related shell commands (python, pip, pytest, etc.), always activate the conda environment first.

Use the following pattern for all shell commands:

```
conda activate BasicTS & <实际命令>
```

For example:
- `conda activate BasicTS & python train.py`
- `conda activate BasicTS & pip install -r requirements.txt`

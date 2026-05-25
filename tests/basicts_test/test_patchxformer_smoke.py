# pylint: disable=wrong-import-position
"""
Smoke tests for PatchXformerForForecasting.

Verifies:
1. Instantiation with typical config (37 features, input_len=96, output_len=12)
2. Output shape is [B, output_len, 1]
3. output_attentions=True returns dict with "prediction" and "attn_weights"
4. output_attentions=False returns just the tensor
5. Error cases (invalid configs raise ValueError)
6. exclude_target_from_input mode (use_asymmetric_attn=True with only process vars
   should fallback to self-attention)
"""

import os
import sys
import types

import pytest
import torch

# Add src to path so basicts submodules can be imported
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, _SRC)

# Provide a minimal EasyDict shim if easydict is not installed
try:
    import easydict  # noqa: F401
except ImportError:
    _easydict_mod = types.ModuleType("easydict")

    class _EasyDict(dict):
        """Minimal EasyDict shim for testing."""
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            del self[name]

    _easydict_mod.EasyDict = _EasyDict
    sys.modules["easydict"] = _easydict_mod

# Provide a minimal easytorch shim to prevent import errors from basicts.__init__
try:
    import easytorch  # noqa: F401
except ImportError:
    _easytorch_mod = types.ModuleType("easytorch")
    sys.modules["easytorch"] = _easytorch_mod
    # Add sub-modules that basicts might try to import
    for submod in ["easytorch.config", "easytorch.core", "easytorch.launcher",
                   "easytorch.utils", "easytorch.device"]:
        sys.modules[submod] = types.ModuleType(submod)

# Now import the PatchXformer modules
from basicts.configs.model_config import BasicTSModelConfig  # noqa: E402
from basicts.models.PatchXformer.config.patchxformer_config import PatchXformerConfig  # noqa: E402
from basicts.models.PatchXformer.arch.layers import DualAxisBlock, VariableImportanceGate  # noqa: E402
from basicts.models.PatchXformer.arch.patchxformer_arch import (  # noqa: E402
    PatchXformerBackbone,
    PatchXformerForForecasting,
)

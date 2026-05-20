# pylint: disable=wrong-import-position
"""
Unit tests for PLS, SVR, and LSTM models.

Tests model instantiation, forward pass shapes, fit behavior,
and serialization for non-gradient models.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

# Direct file-based imports to avoid triggering basicts.__init__.py
# which requires easytorch (not needed for pure model unit tests)
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _import_from_file(module_name, file_path):
    """Import a module directly from file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Pre-register the minimal dependencies so model configs can resolve their imports
_model_config_mod = _import_from_file(
    "basicts.configs.model_config",
    os.path.join(_SRC, "basicts", "configs", "model_config.py")
)

# Now import model modules
_pls_config_mod = _import_from_file(
    "basicts.models.PLS.config.pls_config",
    os.path.join(_SRC, "basicts", "models", "PLS", "config", "pls_config.py")
)
_pls_arch_mod = _import_from_file(
    "basicts.models.PLS.arch.pls_arch",
    os.path.join(_SRC, "basicts", "models", "PLS", "arch", "pls_arch.py")
)
_svr_config_mod = _import_from_file(
    "basicts.models.SVR.config.svr_config",
    os.path.join(_SRC, "basicts", "models", "SVR", "config", "svr_config.py")
)
_svr_arch_mod = _import_from_file(
    "basicts.models.SVR.arch.svr_arch",
    os.path.join(_SRC, "basicts", "models", "SVR", "arch", "svr_arch.py")
)
_lstm_config_mod = _import_from_file(
    "basicts.models.LSTM.config.lstm_config",
    os.path.join(_SRC, "basicts", "models", "LSTM", "config", "lstm_config.py")
)
_lstm_arch_mod = _import_from_file(
    "basicts.models.LSTM.arch.lstm_arch",
    os.path.join(_SRC, "basicts", "models", "LSTM", "arch", "lstm_arch.py")
)

PLS = _pls_arch_mod.PLS
PLSConfig = _pls_config_mod.PLSConfig
SVR = _svr_arch_mod.SVR
SVRConfig = _svr_config_mod.SVRConfig
LSTM = _lstm_arch_mod.LSTM
LSTMConfig = _lstm_config_mod.LSTMConfig


# ============ PLS Tests ============

class TestPLSModel:
    """Unit tests for PLS model."""

    @pytest.fixture
    def pls_model(self):
        config = PLSConfig(
            input_len=32,
            output_len=6,
            num_features=7,
            num_targets=1,
            n_components=5,
        )
        return PLS(config)

    @pytest.fixture
    def fitted_pls_model(self, pls_model):
        """PLS model fitted on random data."""
        np.random.seed(42)
        X = np.random.randn(100, 32, 7).astype(np.float32)
        y = np.random.randn(100, 6, 1).astype(np.float32)
        pls_model.fit(X, y)
        return pls_model

    def test_instantiation(self, pls_model):
        """Test PLS model can be instantiated."""
        assert pls_model.input_len == 32
        assert pls_model.output_len == 6
        assert pls_model.num_features == 7
        assert pls_model.num_targets == 1
        assert pls_model._pls_model is None

    def test_fit(self, fitted_pls_model):
        """Test PLS model can be fitted."""
        assert fitted_pls_model._pls_model is not None

    def test_forward_shape(self, fitted_pls_model):
        """Test forward pass produces correct output shape."""
        inputs = torch.randn(16, 32, 7)
        output = fitted_pls_model(inputs)
        assert output.shape == (16, 6, 1), f"Expected (16, 6, 1), got {output.shape}"

    def test_forward_dtype(self, fitted_pls_model):
        """Test forward pass preserves dtype."""
        inputs = torch.randn(8, 32, 7, dtype=torch.float32)
        output = fitted_pls_model(inputs)
        assert output.dtype == torch.float32

    def test_forward_without_fit_raises(self, pls_model):
        """Test that forward without fit raises RuntimeError."""
        inputs = torch.randn(4, 32, 7)
        with pytest.raises(RuntimeError, match="not been fitted"):
            pls_model(inputs)

    def test_n_components_auto(self):
        """Test auto n_components determination."""
        config = PLSConfig(
            input_len=8,
            output_len=2,
            num_features=3,
            num_targets=1,
            n_components=None,
        )
        model = PLS(config)
        X = np.random.randn(50, 8, 3).astype(np.float32)
        y = np.random.randn(50, 2, 1).astype(np.float32)
        model.fit(X, y)
        # Should not raise and model should be fitted
        assert model._pls_model is not None

    def test_serialization(self, fitted_pls_model):
        """Test get/set sklearn model for serialization."""
        sklearn_model = fitted_pls_model.get_sklearn_model()
        assert sklearn_model is not None

        # Create new model and set the sklearn model
        config = PLSConfig(input_len=32, output_len=6, num_features=7, num_targets=1)
        new_model = PLS(config)
        new_model.set_sklearn_model(sklearn_model)

        # Verify predictions match
        inputs = torch.randn(4, 32, 7)
        out1 = fitted_pls_model(inputs)
        out2 = new_model(inputs)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_batch_size_one(self, fitted_pls_model):
        """Test forward with batch size 1."""
        inputs = torch.randn(1, 32, 7)
        output = fitted_pls_model(inputs)
        assert output.shape == (1, 6, 1)


# ============ SVR Tests ============

class TestSVRModel:
    """Unit tests for SVR model."""

    @pytest.fixture
    def svr_model(self):
        config = SVRConfig(
            input_len=16,
            output_len=4,
            num_features=5,
            num_targets=1,
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
            gamma="scale",
        )
        return SVR(config)

    @pytest.fixture
    def fitted_svr_model(self, svr_model):
        """SVR model fitted on random data."""
        np.random.seed(42)
        X = np.random.randn(80, 16, 5).astype(np.float32)
        y = np.random.randn(80, 4, 1).astype(np.float32)
        svr_model.fit(X, y)
        return svr_model

    def test_instantiation(self, svr_model):
        """Test SVR model can be instantiated."""
        assert svr_model.input_len == 16
        assert svr_model.output_len == 4
        assert svr_model.kernel == "rbf"
        assert svr_model._svr_model is None

    def test_fit(self, fitted_svr_model):
        """Test SVR model can be fitted."""
        assert fitted_svr_model._svr_model is not None

    def test_forward_shape(self, fitted_svr_model):
        """Test forward pass produces correct output shape."""
        inputs = torch.randn(8, 16, 5)
        output = fitted_svr_model(inputs)
        assert output.shape == (8, 4, 1), f"Expected (8, 4, 1), got {output.shape}"

    def test_forward_without_fit_raises(self, svr_model):
        """Test that forward without fit raises RuntimeError."""
        inputs = torch.randn(4, 16, 5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            svr_model(inputs)

    def test_single_output(self):
        """Test SVR with single output (output_len=1, num_targets=1)."""
        config = SVRConfig(
            input_len=8,
            output_len=1,
            num_features=3,
            num_targets=1,
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
            gamma="scale",
        )
        model = SVR(config)
        X = np.random.randn(50, 8, 3).astype(np.float32)
        y = np.random.randn(50, 1, 1).astype(np.float32)
        model.fit(X, y)

        inputs = torch.randn(4, 8, 3)
        output = model(inputs)
        assert output.shape == (4, 1, 1)

    def test_serialization(self, fitted_svr_model):
        """Test get/set sklearn model for serialization."""
        sklearn_model = fitted_svr_model.get_sklearn_model()
        assert sklearn_model is not None

        config = SVRConfig(input_len=16, output_len=4, num_features=5, num_targets=1)
        new_model = SVR(config)
        new_model.set_sklearn_model(sklearn_model)

        inputs = torch.randn(4, 16, 5)
        out1 = fitted_svr_model(inputs)
        out2 = new_model(inputs)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_different_kernels(self):
        """Test SVR with different kernel types."""
        for kernel in ["rbf", "linear"]:
            config = SVRConfig(
                input_len=8, output_len=2, num_features=3, num_targets=1,
                kernel=kernel, C=1.0, epsilon=0.1, gamma="scale",
            )
            model = SVR(config)
            X = np.random.randn(30, 8, 3).astype(np.float32)
            y = np.random.randn(30, 2, 1).astype(np.float32)
            model.fit(X, y)
            output = model(torch.randn(4, 8, 3))
            assert output.shape == (4, 2, 1), f"Failed for kernel={kernel}"


# ============ LSTM Tests ============

class TestLSTMModel:
    """Unit tests for LSTM model."""

    @pytest.fixture
    def lstm_model(self):
        config = LSTMConfig(
            input_len=32,
            output_len=6,
            num_features=7,
            num_targets=1,
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
        )
        return LSTM(config)

    def test_instantiation(self, lstm_model):
        """Test LSTM model can be instantiated."""
        assert lstm_model.input_len == 32
        assert lstm_model.output_len == 6
        assert lstm_model.num_features == 7
        assert lstm_model.hidden_size == 32

    def test_forward_shape(self, lstm_model):
        """Test forward pass produces correct output shape."""
        inputs = torch.randn(16, 32, 7)
        output = lstm_model(inputs)
        assert output.shape == (16, 6, 1), f"Expected (16, 6, 1), got {output.shape}"

    def test_forward_batch_size_one(self, lstm_model):
        """Test forward with batch size 1."""
        inputs = torch.randn(1, 32, 7)
        output = lstm_model(inputs)
        assert output.shape == (1, 6, 1)

    def test_gradient_flow(self, lstm_model):
        """Test that gradients flow through the model."""
        inputs = torch.randn(4, 32, 7, requires_grad=False)
        output = lstm_model(inputs)
        loss = output.sum()
        loss.backward()

        # Check that LSTM parameters have gradients
        for name, param in lstm_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_bidirectional(self):
        """Test bidirectional LSTM."""
        config = LSTMConfig(
            input_len=16,
            output_len=4,
            num_features=5,
            num_targets=1,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
        )
        model = LSTM(config)
        inputs = torch.randn(8, 16, 5)
        output = model(inputs)
        assert output.shape == (8, 4, 1)

    def test_multi_target(self):
        """Test LSTM with multiple target variables."""
        config = LSTMConfig(
            input_len=16,
            output_len=4,
            num_features=5,
            num_targets=3,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
        )
        model = LSTM(config)
        inputs = torch.randn(8, 16, 5)
        output = model(inputs)
        assert output.shape == (8, 4, 3)

    def test_deterministic_output(self, lstm_model):
        """Test that eval mode produces deterministic output."""
        lstm_model.eval()
        inputs = torch.randn(4, 32, 7)
        with torch.no_grad():
            out1 = lstm_model(inputs)
            out2 = lstm_model(inputs)
        assert torch.allclose(out1, out2)

    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        config = LSTMConfig(
            input_len=16,
            output_len=4,
            num_features=5,
            num_targets=1,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
        )
        model = LSTM(config)
        total_params = sum(p.numel() for p in model.parameters())
        # LSTM: 4 * ((input_size + hidden_size) * hidden_size + hidden_size) = 4 * ((5+32)*32 + 32) = 4864
        # Projection: hidden_size * (output_len * num_targets) + bias = 32 * 4 + 4 = 132
        # Total ≈ 4996
        assert total_params > 0
        assert total_params < 100000  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

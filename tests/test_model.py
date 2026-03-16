"""Tests for CNN-LSTM model."""

import torch
import pytest

from model import CNNLSTMRegressor


class TestCNNLSTMRegressor:
    def test_output_shape(self):
        model = CNNLSTMRegressor(n_features=27)
        x = torch.randn(4, 60, 27)
        out = model(x)
        assert out.shape == (4,)

    def test_single_sample(self):
        model = CNNLSTMRegressor(n_features=10, lstm_hidden=32)
        x = torch.randn(1, 30, 10)
        out = model(x)
        assert out.shape == (1,)

    def test_different_configs(self):
        model = CNNLSTMRegressor(
            n_features=5,
            cnn_channels=32,
            kernel=3,
            lstm_hidden=48,
            lstm_layers=1,
            dropout=0.1,
        )
        x = torch.randn(2, 20, 5)
        out = model(x)
        assert out.shape == (2,)

    def test_gradients_flow(self):
        model = CNNLSTMRegressor(n_features=10)
        x = torch.randn(2, 30, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from crimson_quant.config import Config


class TestConfigDefaults(unittest.TestCase):
    def test_default_epochs(self):
        cfg = Config()
        assert cfg.epochs == 300

    def test_default_patience(self):
        cfg = Config()
        assert cfg.patience == 30


class TestConfigLoad(unittest.TestCase):
    def _write_config(self, tmp_dir, data):
        path = os.path.join(tmp_dir, "config.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_load_epochs_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(tmp, {"epochs": 500})
            with patch("crimson_quant.config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.epochs == 500

    def test_load_patience_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(tmp, {"patience": 50})
            with patch("crimson_quant.config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.patience == 50

    def test_unknown_fields_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(tmp, {"epochs": 100, "unknown_field": 99})
            with patch("crimson_quant.config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.epochs == 100
        assert not hasattr(cfg, "unknown_field")

    def test_missing_json_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            # no config.json written
            with patch("crimson_quant.config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.epochs == 300
        assert cfg.patience == 30


class TestConfigLookbackDefault(unittest.TestCase):
    def test_default_lookback(self):
        cfg = Config()
        assert cfg.lookback == 60

    def test_load_lookback_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w") as f:
                json.dump({"lookback": 90}, f)
            with patch("crimson_quant.config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.lookback == 90

    def test_lookback_not_overridden_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w") as f:
                json.dump({"epochs": 100}, f)
            with patch("crimson_quant.config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.lookback == 60


class TestInteractiveConfigLookback(unittest.TestCase):
    def _run_interactive(self, inputs, tmp_dir):
        """Helper: run _interactive_config with mocked stdin and config path."""
        from crimson_quant.config import _interactive_config
        with patch("builtins.input", side_effect=inputs), \
             patch("crimson_quant.config.os.path.dirname", return_value=tmp_dir):
            _interactive_config()
        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path) as f:
            return json.load(f)

    def test_lookback_written_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            # inputs: ticker, start, end, quantile, lookback=90, epochs, patience
            inputs = ["AAPL", "", "", "", "90", "", ""]
            saved = self._run_interactive(inputs, tmp)
        assert saved["lookback"] == 90

    def test_lookback_defaults_kept_on_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = ["AAPL", "", "", "", "", "", ""]
            saved = self._run_interactive(inputs, tmp)
        assert saved["lookback"] == 60

    def test_lookback_invalid_value_keeps_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            # invalid "abc" then empty (keep default)
            inputs = ["AAPL", "", "", "", "abc", "", "", ""]
            saved = self._run_interactive(inputs, tmp)
        assert saved["lookback"] == 60

    def test_lookback_zero_rejected_keeps_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = ["AAPL", "", "", "", "0", "", ""]
            saved = self._run_interactive(inputs, tmp)
        assert saved["lookback"] == 60

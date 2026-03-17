import json
import os
import tempfile
import unittest
from unittest.mock import patch

from config import Config


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
            with patch("config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.epochs == 500

    def test_load_patience_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(tmp, {"patience": 50})
            with patch("config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.patience == 50

    def test_unknown_fields_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(tmp, {"epochs": 100, "unknown_field": 99})
            with patch("config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.epochs == 100
        assert not hasattr(cfg, "unknown_field")

    def test_missing_json_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            # no config.json written
            with patch("config.os.path.dirname", return_value=tmp):
                cfg = Config.load()
        assert cfg.epochs == 300
        assert cfg.patience == 30

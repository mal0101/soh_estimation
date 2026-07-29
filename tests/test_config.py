"""Tests for src/utils/config.py."""

from pathlib import Path

import pytest

from src.utils.config import Config

CONFIG_PATH = Path("config/default.yaml")


class TestConfigFromYaml:
    """Tests for Config.from_yaml class method."""

    def test_loads_without_error(self):
        config = Config.from_yaml(CONFIG_PATH)
        assert isinstance(config, Config)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent.yaml")


class TestConfigAccess:
    """Tests for dot-notation and dict-style access."""

    def setup_method(self):
        self.config = Config.from_yaml(CONFIG_PATH)

    def test_dot_access_rf_trials(self):
        assert self.config.models.classical.rf.n_trials == 100

    def test_dict_access_rf_trials(self):
        assert self.config["models"]["classical"]["rf"]["n_trials"] == 100

    def test_get_dotted_path(self):
        assert self.config.get("preprocessing.savgol.window_length") == 51

    def test_get_missing_key_returns_default(self):
        assert self.config.get("nonexistent.key", 42) == 42

    def test_get_nested_returns_config(self):
        val = self.config.get("models.classical")
        assert isinstance(val, Config)

    def test_numpy_array_access(self):
        cells = self.config.data.nasa_pcoe.cells
        assert cells == ["B0005", "B0006", "B0007", "B0018"]

    def test_to_dict(self):
        d = self.config.to_dict()
        assert isinstance(d, dict)
        assert "models" in d


class TestConfigValues:
    """Tests that specific config values match expected defaults."""

    def setup_method(self):
        self.config = Config.from_yaml(CONFIG_PATH)

    def test_svr_c_is_two_element(self):
        c = self.config["models"]["classical"]["svr"]["param_space"]["C"]
        assert len(c) == 2

    def test_dl_learning_rates_are_two_element(self):
        for model in ["lstm", "cnn", "transformer"]:
            lr = self.config["models"]["dl"][model]["param_space"]["learning_rate"]
            assert len(lr) == 2, f"{model} learning_rate should be 2 elements"

    def test_svr_c_no_log_string(self):
        c = self.config["models"]["classical"]["svr"]["param_space"]["C"]
        assert "log" not in c, "SVR C param should not contain 'log' string"

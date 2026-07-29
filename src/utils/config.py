"""Centralized configuration loader.

Loads YAML config files and provides dot-notation access to nested keys.
All project hyperparameters, paths, and settings flow through this module
to ensure a single source of truth and full reproducibility.
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Hierarchical configuration with dot-notation access.

    Example::

        cfg = Config.from_yaml("config/default.yaml")
        lr = cfg.models.dl.learning_rate
        cells = cfg.data.nasa_pcoe.cells
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Config instance with nested attribute access.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        try:
            value = self._data[key]
        except KeyError as err:
            raise AttributeError(f"Config has no attribute '{key}'") from err
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def to_dict(self) -> dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a fallback default.

        Args:
            key: Dot-separated key path (e.g., 'models.dl.batch_size').
            default: Value to return if key is not found.

        Returns:
            The configuration value, or default if not found.
        """
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        if isinstance(value, dict):
            return Config(value)
        return value

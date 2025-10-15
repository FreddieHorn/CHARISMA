import toml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigSection:
    def __init__(self, data: Dict[str, Any]):
        self._build_structure(data)

    def _build_structure(self, data: Dict[str, Any]) -> None:
        """Recursively build the configuration structure"""
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name: str) -> None:
        """Handle missing attributes gracefully"""
        return None

    def __repr__(self) -> str:
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})


class Config:
    def __init__(self, toml_file: str = "charisma/default.toml"):
        self._toml_file = toml_file
        self._load_config()

    def _load_config(self) -> None:
        if not Path(self._toml_file).exists():
            raise FileNotFoundError(f"Configuration file {self._toml_file} not found")

        config_data = toml.load(self._toml_file)
        self._build_structure(config_data)

    def _build_structure(self, data: Dict[str, Any]) -> None:
        for section, values in data.items():
            parts = section.split(".")
            current = self

            for part in parts[:-1]:
                if not hasattr(current, part):
                    setattr(current, part, ConfigSection({}))
                current = getattr(current, part)

            if isinstance(values, dict):
                setattr(current, parts[-1], ConfigSection(values))
            else:
                setattr(current, parts[-1], values)

    def get(self, path: str, default: Any = None) -> Any:
        current = self
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return default
        return current

    def reload(self) -> None:
        self._load_config()

    def __repr__(self) -> str:
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})


config = Config()

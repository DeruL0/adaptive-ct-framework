from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file(config_path: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in stack:
        chain = " -> ".join(str(value) for value in (*stack, config_path))
        raise ValueError(f"Config inheritance cycle: {chain}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} did not contain a mapping.")
    parent = data.pop("extends", None)
    if parent is not None:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        data = _deep_merge(_load_config_file(parent_path, (*stack, config_path)), data)
    return data


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    data = _load_config_file(config_path, ())
    data["_config_path"] = str(config_path.resolve())
    data["_config_dir"] = str(config_path.resolve().parent)
    return data


def get_nested(config: Dict[str, Any], key: str, default=None):
    cur: Any = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

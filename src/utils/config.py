"""Load configs/*.yaml as the single source of hyperparameters.

The project now keeps one config: configs/config.yaml.

Base inheritance is still supported (a config with a 'base:' key is deep-merged
on top of its parent), but no config currently depends on it.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/config.yaml")         # dict
"""
import os
import copy

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "configs/config.yaml"):
    """Read a YAML config (always UTF-8).  If it has a 'base' key, deep-merge
    that base first, then apply this file's own keys on top."""
    with open(_resolve(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None) if isinstance(cfg, dict) else None
    if base:
        parent = load_config(base)          # recursive; resolves relative to project root
        cfg = _deep_merge(parent, cfg)
    return cfg


def get(cfg, dotted_key, default=None):
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

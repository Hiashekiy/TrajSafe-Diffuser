"""Load configs/*.yaml as the single source of hyperparameters.

The project keeps one config: configs/config.yaml (training, data,
model, loss and the inference-time `alm` section used by the dashboard).

Base inheritance is still supported (a config with a 'base:' key is deep-merged
on top of its parent), but no shipped config depends on it.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/config.yaml")     # dict
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


def num_controls(cfg, default=32):
    """The ONE number of B-spline control points, read from the config.

    ``bspline.num_controls`` (the codec) and ``model.num_controls`` (the
    network / diffusion state) must agree when both are present, so changing C
    is a one-line edit in configs/config.yaml.  Nothing in the code hard-codes
    32 any more.
    """
    bs = get(cfg, "bspline.num_controls")
    md = get(cfg, "model.num_controls")
    if bs is not None and md is not None and int(bs) != int(md):
        raise ValueError(
            "config mismatch: bspline.num_controls=%s != model.num_controls=%s"
            % (bs, md))
    value = bs if bs is not None else md
    if value is None:
        value = default
    value = int(value)
    if value < 2:
        raise ValueError("num_controls must be >= 2, got %d" % value)
    return value


def curve_points(cfg, default=128):
    """Decoded curve sampling density (``bspline.curve_points``)."""
    value = get(cfg, "bspline.curve_points")
    if value is None:
        value = get(cfg, "model.horizon")
    return int(default if value is None else value)


def num_safety_queries(cfg, default=None):
    """Number of Skeleton / ellipse geometry queries (``model.num_safety_queries``).

    Defaults to ``topology.candidate_points`` because the safety queries are
    exactly the selected Skeleton tokens; the decoded curve density is only a
    last-resort fallback (conflating the two is what this refactor removes).
    """
    value = get(cfg, "model.num_safety_queries")
    if value is None:
        value = get(cfg, "topology.candidate_points")
    if value is None:
        value = get(cfg, "model.horizon")
    return int(default if value is None else value)


"""Model factory: pick the framework from ``model.framework`` (default = legacy).

    cfg["model"]["framework"] == "v5"  ->  TrajSafePlannerV5 (new single-path chain)
    anything else / missing            ->  TrajSafePlanner   (legacy, q_coarse chain)

Everything else (checkpoint loading, samplers, scripts) goes through this helper,
so adding the new framework never changes how an old checkpoint is built.
"""

from __future__ import annotations

__all__ = ["build_model", "framework_of"]


def framework_of(cfg) -> str:
    model_cfg = dict((cfg or {}).get("model") or {})
    return str(model_cfg.get("framework", "legacy")).strip().lower()


def build_model(cfg, model_cfg=None):
    """Build the planner for ``cfg``; ``model_cfg`` overrides ``cfg['model']``."""
    from .trajsafe import TrajSafePlanner

    model_cfg = dict(model_cfg if model_cfg is not None else
                     ((cfg or {}).get("model") or {}))
    fw = str(model_cfg.get("framework", "legacy")).strip().lower()
    if fw in ("v5", "single_path", "singe_path"):
        from .trajsafe.planner_v5 import TrajSafePlannerV5
        return TrajSafePlannerV5(model_cfg, (cfg or {}).get("ellipse_label"),
                                 (cfg or {}).get("bspline"))
    return TrajSafePlanner(model_cfg, (cfg or {}).get("ellipse_label"),
                           (cfg or {}).get("bspline"))

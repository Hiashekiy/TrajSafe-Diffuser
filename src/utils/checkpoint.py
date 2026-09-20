"""Checkpoint save/load utilities.

Besides the plain ``torch`` save/load there is a *pre-refactor aware* loader:

* :func:`detect_architecture` inspects the stored tensors and reports whether a
  checkpoint was written by the control-token model (this refactor) or by the
  legacy curve-token model (the network running on the decoded 128-point curve).
  The old checkpoints stay fully usable: ``sample.py`` / ``evaluate.py`` switch
  to the legacy forward chain automatically (``--arch auto``).
* :func:`load_state_dict_flexible` loads a state dict into a model even when a
  few tensors changed shape (e.g. a different control count C, or a different
  relative-bias table length): parameters are truncated / zero-padded along the
  leading axis, everything else is reported.  Missing modules introduced by the
  refactor (safety query head, safety->control cross attention) simply keep
  their fresh initialisation - the cross attention is zero-initialised, so it
  starts as the identity and does not perturb the loaded network.
"""

import os

import torch
import torch.nn as nn

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "extract_state_dict",
    "load_state_dict_flexible",
    "detect_architecture",
    "resolve_architecture",
    "load_model",
    "ARCH_CONTROL_SPACE",
    "ARCH_LEGACY_CURVE",
]

ARCH_CONTROL_SPACE = "control_space"
ARCH_LEGACY_CURVE = "legacy_curve"

# tensors that only exist in the refactored (control-token) model
_CONTROL_ONLY_PREFIXES = (
    "safety_query_head.",
    "safety_cross_attention.",
)


def save_checkpoint(path, model, optimizer=None, epoch=None, cfg=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"model_state": model.state_dict()}
    if optimizer is not None:
        data["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch
    if cfg is not None:
        data["config"] = cfg
    if extra:
        data.update(extra)
    torch.save(data, path)


def extract_state_dict(data) -> dict:
    """Accept a full checkpoint dict or a bare state dict."""
    if isinstance(data, dict) and "model_state" in data:
        return data["model_state"]
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


def detect_architecture(state) -> str:
    """``control_space`` if the checkpoint has the new modules, else legacy."""
    state = extract_state_dict(state)
    keys = list(state.keys())
    for key in keys:
        for prefix in _CONTROL_ONLY_PREFIXES:
            if key.startswith(prefix):
                return ARCH_CONTROL_SPACE
    return ARCH_LEGACY_CURVE


def _adapt_tensor(tensor: torch.Tensor, target: torch.Tensor):
    """Truncate / zero-pad ``tensor`` along dim 0 to match ``target``."""
    if tensor.dim() != target.dim():
        return None
    if tensor.shape[1:] != target.shape[1:]:
        return None
    n, m = int(tensor.shape[0]), int(target.shape[0])
    out = tensor.new_zeros(target.shape)
    keep = min(n, m)
    if keep > 0:
        out[:keep] = tensor[:keep]
    return out


def load_state_dict_flexible(model, state, strict_shape: bool = False,
                             verbose: bool = True, tag: str = ""):
    """``load_state_dict`` that tolerates refactor-induced shape changes.

    Returns a report dict with ``mapped`` / ``adapted`` / ``missing`` /
    ``unexpected`` keys.  ``strict_shape=False`` never raises for a handful of
    shape changes; it reports them instead.
    """
    state = extract_state_dict(state)
    state = {str(k).replace("module.", "", 1)
             if str(k).startswith("module.") else str(k): v
             for k, v in state.items()}
    own = model.state_dict()
    report = {"mapped": [], "adapted": [], "missing": [],
              "unexpected": [], "arch": detect_architecture(state)}
    new_state = {}
    for key, target in own.items():
        if key not in state:
            report["missing"].append(key)
            continue
        value = state[key]
        if not torch.is_tensor(value):
            report["missing"].append(key)
            continue
        if tuple(value.shape) == tuple(target.shape):
            new_state[key] = value.to(dtype=target.dtype)
            report["mapped"].append(key)
            continue
        adapted = _adapt_tensor(value.to(dtype=target.dtype), target)
        if adapted is None:
            if strict_shape:
                raise RuntimeError(
                    "shape mismatch for %s: checkpoint %s vs model %s"
                    % (key, tuple(value.shape), tuple(target.shape)))
            report["missing"].append(key)
            continue
        new_state[key] = adapted
        report["adapted"].append("%s %s->%s" % (key, tuple(value.shape),
                                                tuple(target.shape)))
    for key in state:
        if key not in own:
            report["unexpected"].append(key)
    model.load_state_dict(new_state, strict=False)
    if verbose:
        print("[ckpt]%s loaded %d tensors (%d shape-adapted, %d fresh, "
              "%d dropped) arch=%s"
              % (" " + tag if tag else "", len(report["mapped"]),
                 len(report["adapted"]), len(report["missing"]),
                 len(report["unexpected"]), report["arch"]), flush=True)
        for key in report["adapted"]:
            print("[ckpt]   adapted %s" % key, flush=True)
        for key in report["missing"][:20]:
            print("[ckpt]   fresh   %s" % key, flush=True)
        if len(report["missing"]) > 20:
            print("[ckpt]   ... %d more fresh tensors"
                  % (len(report["missing"]) - 20), flush=True)
    return report


def load_checkpoint(path, model, optimizer=None, map_location=None,
                    flexible: bool = True, verbose: bool = True):
    data = torch.load(path, map_location=map_location, weights_only=False)
    if flexible:
        load_state_dict_flexible(model, data, verbose=verbose,
                                 tag=os.path.basename(str(path)))
    else:
        model.load_state_dict(extract_state_dict(data))
    if optimizer is not None and isinstance(data, dict) \
            and "optimizer_state" in data:
        try:
            optimizer.load_state_dict(data["optimizer_state"])
        except ValueError as exc:                                  # pragma: no cover
            print("[ckpt] optimizer state ignored (%s)" % exc, flush=True)
    return data


def resolve_architecture(model_cfg, state, arch: str = "auto") -> dict:
    """Copy ``model_cfg`` with ``control_space`` resolved from ``arch``.

    ``arch='auto'`` (the default everywhere) picks the forward chain from the
    checkpoint contents, so an existing pre-refactor checkpoint keeps working
    without any manual flag.
    """
    cfg = dict(model_cfg or {})
    mode = str(arch or "auto").strip().lower()
    if mode == "auto":
        if state is None:
            return cfg
        cfg["control_space"] = detect_architecture(state) == ARCH_CONTROL_SPACE
        return cfg
    if mode in ("control", "control_space", "ctrl", "new"):
        cfg["control_space"] = True
    elif mode in ("legacy", "legacy_curve", "curve", "old"):
        cfg["control_space"] = False
    else:
        raise ValueError("unknown architecture %r (use auto/control/legacy)"
                         % arch)
    return cfg


def load_model(cfg, ckpt=None, arch: str = "auto", device="cpu",
               verbose: bool = True):
    """Build a :class:`TrajSafePlanner` and load ``ckpt`` into it.

    Returns ``(model, checkpoint_dict_or_None, report_or_None)``.  The
    architecture (control-token chain vs the legacy curve-token chain) is taken
    from the checkpoint unless ``arch`` forces it, and shape changes introduced
    by the refactor are absorbed by :func:`load_state_dict_flexible`.
    """
    from ..models.trajsafe import TrajSafePlanner

    data = None
    if isinstance(ckpt, str):
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
    elif isinstance(ckpt, dict):
        data = ckpt
    state = extract_state_dict(data) if data is not None else None
    model_cfg = resolve_architecture(cfg.get("model"), state, arch)
    model = TrajSafePlanner(model_cfg, cfg.get("ellipse_label"),
                            cfg.get("bspline"))
    report = None
    if state is not None:
        report = load_state_dict_flexible(
            model, state, verbose=verbose,
            tag=os.path.basename(str(ckpt)) if isinstance(ckpt, str) else "")
    model.to(device)
    return model, data, report

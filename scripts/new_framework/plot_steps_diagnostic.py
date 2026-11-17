"""Per-step diagnostic figure: prediction, ellipses, ALM correction, searched path.

For each requested test sample the script replays the exact 4-step pipeline
(step 1 without path -> PDG-Focal A* -> steps 2..4 with the local-corridor ALM
closed loop) and draws, per step:

    * occupancy (grey)                       * searched guidance path (thick blue)
    * the network's raw x0 (orange)          * ALM-corrected q0_safe (green dashed, if any)
    * the 64 predicted safety ellipses (thin cyan, every 4th drawn)

Panel titles carry the measured numbers: collision before -> after, whether ALM
ran and whether it passed the global validation, and the correction size.

    python scripts/new_framework/plot_steps_diagnostic.py \
        --samples 3009 3011 1001 --out outputs/v5_eval/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
    __file__))))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate
from src.diffusion.alm_local import (alm_closed_loop, collision_intervals,
                                    collision_mask)
from src.diffusion.sampler import pick_times
from src.diffusion.sampler_v5 import _build_safe_grids, _search_once, ddim_step_v5
from src.diffusion.schedule import NoiseSchedule
from src.geometry.convex_region import EllipseRegionBuilder
from src.geometry.safety_corridor import halfspaces_to_vertices
from src.models.factory import build_model
from src.utils.config import load_config

SCENE_TO_METER = 80.0


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ellipse_poly(cx, cy, a, b, th, n=40):
    t = np.linspace(0, 2 * np.pi, n)
    ex, ey = a * np.cos(t), b * np.sin(t)
    ct, st = np.cos(th), np.sin(th)
    return cx + ct * ex - st * ey, cy + st * ex + ct * ey


def replay(model, sched, ds, pos, cfg, device, steps, seed=0):
    torch.manual_seed(int(seed))          # figures must be reproducible
    np.random.seed(int(seed))
    batch = make_collate(ds)([ds[pos]])
    cond = batch["cond"].to(device)
    occ = batch["occupancy"].to(device)
    heading = batch.get("heading")
    heading = heading.to(device) if heading is not None else None
    hv = batch.get("heading_valid")
    hv = hv.to(device) if hv is not None else None
    grids, spec, _ = _build_safe_grids(occ, cfg)
    B, C = 1, int(model.num_controls)
    T = int(sched.num_timesteps)
    times = pick_times(T, steps) or list(range(T))
    pairs = [(times[i], times[i - 1]) for i in range(len(times) - 1, 0, -1)]
    pairs.append((times[0], -1))

    q = torch.randn(B, C, 2, device=device)
    q[:, 0] = cond[:, 0]
    q[:, -1] = cond[:, 1]
    active_path, q0_safe, lam_prev = None, None, None
    fb_delta = torch.zeros(B, C, 2, device=device)
    fb_valid = torch.zeros(B, dtype=torch.bool, device=device)
    recs = []
    with torch.no_grad():
        for i, (t, s) in enumerate(pairs):
            tb = torch.full((B,), int(t), device=device, dtype=torch.long)
            ab = sched.sqrt_alphas_cumprod[int(t)].expand(B).to(device)
            out = model(q, occ, cond, tb, ab, active_path=active_path,
                        feedback_delta=(fb_delta if i >= 2 else None),
                        feedback_valid=(fb_valid if i >= 2 else None),
                        heading=heading, heading_valid=hv)
            raw = out["final"][0].detach().cpu().numpy()
            hit, _ = collision_mask(occ[0], raw)
            rec = {"step": i + 1, "t": int(t), "raw": raw,
                   "ell": {k: v[0].detach().cpu().numpy()
                           for k, v in (out["ellipse"] or {}).items()},
                   "path": None if active_path is None else
                   active_path[0].detach().cpu().numpy(),
                   "coll_before": float(hit.mean()), "safe": None,
                   "alm_ran": False, "alm_ok": False, "corr": 0.0,
                   "reason": "", "safe": None, "coll_after": None,
                   "safe_hit": None}
            if i == 0:
                p, st = _search_once(grids, spec, cond, out["final"].detach(),
                                     cfg, 0)
                # the same search without centering, for the before/after view
                cfg_raw = {**cfg, "grid_search": {**dict(cfg.get("grid_search")
                                                         or {})}}
                cfg_raw["grid_search"]["center_path"] = {"enabled": False}
                try:
                    p_raw, _ = _search_once(grids, spec, cond,
                                            out["final"].detach(), cfg_raw, 0)
                except Exception:
                    p_raw = None
                active_path = torch.from_numpy(p)[None].to(device)
                rec["path"] = p
                rec["path_raw"] = p_raw
                rec["search"] = st
            else:
                if bool(hit.any()):
                    q0s, lam_prev, info = alm_closed_loop(
                        model, occ, cond, out["control"], dict(out["ellipse"]),
                        cfg, gamma=active_path, lam=lam_prev)
                    rec["alm_ran"] = bool(info.get("alm"))
                    rec["alm_ok"] = bool(info.get("alm_ok"))
                    rec["reason"] = (info.get("reason") or "")[:40]
                    rec["corr"] = float((q0s - out["control"]).norm(dim=-1).mean())
                    rec["cells"] = info.get("cells") or []
                    rec["inserted"] = info.get("inserted_ellipses")
                    rec["corridor_valid"] = info.get("corridor_valid")
                    if info.get("alm"):
                        rec["safe"] = model.bspline.decode_controls(q0s)[0] \
                            .detach().cpu().numpy()
                        hit_s, _ = collision_mask(occ[0], rec["safe"])
                        rec["coll_after"] = float(hit_s.mean())
                        rec["safe_hit"] = rec["safe"][hit_s]
                    if rec["alm_ok"]:
                        active_path = model.bspline.decode_controls(q0s)
                        q0_safe = q0s
                        fb_delta = (q0s - out["control"]).detach()
                        fb_valid = torch.ones(B, dtype=torch.bool, device=device)
            recs.append(rec)
            q0_ddim = (q0_safe if (i > 0 and q0_safe is not None
                                   and rec["alm_ok"]) else out["control"])
            q = ddim_step_v5(q, q0_ddim, tb, s, sched, cond)
    return batch, recs


def plot(plt, batch, recs, sample_id, out_path, stride=4, cfg=None):
    occ = batch["occupancy"][0, 0].numpy()
    occ_t = batch["occupancy"][0:1]
    gt = batch["pos"][0].numpy()
    cond = batch["cond"][0].numpy()
    n = len(recs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.4), squeeze=False)
    px = lambda p: (np.asarray(p) + 1) / 2 * 256 - 0.5            # noqa: E731
    for ax, r in zip(axes[0], recs):
        ax.imshow(occ, origin="lower", cmap="gray_r", alpha=0.45,
                  interpolation="nearest")
        ax.plot(*px(gt).T, color="#2ca02c", lw=1.0, alpha=0.8, label="GT")
        if r.get("path_raw") is not None:
            pr = px(r["path_raw"])
            ax.plot(pr[:, 0], pr[:, 1], color="#7f7f7f", lw=1.0, ls=":",
                    alpha=0.9, label="A* path (before centering)")
        if r["path"] is not None:
            p = px(r["path"])
            ax.plot(p[:, 0], p[:, 1], color="#1f77b4", lw=2.6, alpha=0.9,
                    label="searched path (centered)")
        e = r["ell"]
        if e:
            cx, cy = px(np.stack([e["center"][:, 0], e["center"][:, 1]], -1)).T
            ax.scatter(cx, cy, s=3, c="#17becf", alpha=0.75, zorder=3)
            for k in range(0, len(cx), stride):
                a = float(e["a"][k]) * 128.0       # scene -> cells
                b = float(e["b"][k]) * 128.0
                ex, ey = ellipse_poly(cx[k], cy[k], a, b, float(e["theta"][k]))
                ax.plot(ex, ey, color="#17becf", lw=0.7, alpha=0.75)
        raw = px(r["raw"])
        ax.plot(raw[:, 0], raw[:, 1], color="#ff7f0e", lw=1.4, alpha=0.85,
                label="prediction x0 (raw)")
        hit, _ = collision_mask(batch["occupancy"][0], r["raw"])
        ax.scatter(raw[hit, 0], raw[hit, 1], s=8, c="#d62728", zorder=6,
                   label="collision")
        # ---- per-query convex cells (drawn even if the corridor fails) -----
        ell_d = r["ell"] or {}
        if ell_d and "center" in ell_d and "shape4" in ell_d:
            from src.diffusion.alm_local import _REGION_KEYS
            alm_cfg = dict(((cfg or {}).get("alm") or {}))
            bcfg = {k: alm_cfg[k] for k in _REGION_KEYS if k in alm_cfg}
            try:
                bld = EllipseRegionBuilder(occ_t.float(), bcfg)
                A_, b_, m_, v_ = bld.build_from_ellipse(
                    torch.as_tensor(ell_d["center"])[None],
                    torch.as_tensor(ell_d["shape4"])[None])
                A0 = A_[0].detach().cpu().numpy()
                b0 = b_[0].detach().cpu().numpy()
                m0 = m_[0].detach().cpu().numpy().astype(bool)
                v0 = v_[0].detach().cpu().numpy().astype(bool)
                r["cells_built"] = int(v0.sum())
                r["cells_invalid"] = int((~v0).sum())
                # mark every query whose convex region is degenerate: this is
                # exactly what `invalid_base_region` reports, and it is why the
                # corridor cannot be built at all.  The ellipse itself is still
                # drawn (cyan), the X says "this cell is empty".
                bad_q = [qi for qi in range(A0.shape[0]) if not v0[qi]]
                if bad_q and ell_d.get("center") is not None:
                    cen = px(np.asarray(ell_d["center"])[bad_q])
                    ax.scatter(cen[:, 0], cen[:, 1], s=22, marker="x",
                               c="#d62728", zorder=9,
                               label="degenerate cell (%d)" % len(bad_q))
                for qi in range(A0.shape[0]):
                    if m0[qi].sum() < 3:
                        continue
                    poly = halfspaces_to_vertices(A0[qi][m0[qi]], b0[qi][m0[qi]])
                    if poly is None or len(poly) < 3:
                        continue
                    pp = px(poly)
                    if v0[qi]:
                        ax.fill(pp[:, 0], pp[:, 1], color="#4c72b0", alpha=0.055,
                                lw=0.0, zorder=1)
                        ax.plot(pp[:, 0], pp[:, 1], color="#4c72b0", lw=0.4,
                                alpha=0.30, zorder=2,
                                label="convex cell (valid)")
                    else:
                        ax.fill(pp[:, 0], pp[:, 1], color="#d62728", alpha=0.16,
                                lw=0.0, zorder=1)
                        ax.plot(pp[:, 0], pp[:, 1], color="#d62728", lw=0.5,
                                alpha=0.55, zorder=2,
                                label="convex cell (invalid)")
            except Exception as exc:
                r["cell_error"] = repr(exc)[:60]

        # ---- the corridor the ALM actually received (may be empty) ---------
        for poly in (r.get("cells") or [[]])[0]:
            pp = px(poly)
            ax.fill(pp[:, 0], pp[:, 1], color="#4c72b0", alpha=0.055, lw=0.0,
                    zorder=1)
            ax.plot(pp[:, 0], pp[:, 1], color="#4c72b0", lw=0.4, alpha=0.30,
                    zorder=2, label="corridor cell (convex)")
        if r["safe"] is not None:
            sf = px(r["safe"])
            lab = ("ALM corrected (PASS)" if r["alm_ok"]
                   else "ALM corrected (REJECTED)")
            ax.plot(sf[:, 0], sf[:, 1], color="#9467bd", lw=2.4, ls="--",
                    zorder=7, label=lab)
            if r.get("safe_hit") is not None and len(r["safe_hit"]):
                sh = px(r["safe_hit"])
                ax.scatter(sh[:, 0], sh[:, 1], s=14, marker="x", c="#9467bd",
                           zorder=7)
        s_, g_ = px(cond[0]), px(cond[1])
        ax.scatter(*s_, marker="*", s=190, c="#17a34a", edgecolors="k", zorder=8)
        ax.scatter(*g_, marker="*", s=190, c="#e31a1c", edgecolors="k", zorder=8)
        ttl = ["step %d  t=%d" % (r["step"], r["t"]),
               "coll %.3f%s" % (r["coll_before"],
                                "" if not r["alm_ran"] else
                                (" -> ALM %s" % ("PASS" if r["alm_ok"] else "REJECT"))),
               ("corr %.2f m  |  coll after %.3f  |  cells %d/%d invalid (+%s ins)" % (
                   r["corr"] * SCENE_TO_METER,
                   r["coll_after"] if r["coll_after"] is not None else -1,
                   r.get("cells_invalid", 0), r.get("cells_built", 0)
                   + r.get("cells_invalid", 0), r.get("inserted")))
               if r["alm_ran"] else
               ("search -> %s (exp %s)" % (r.get("search", {}).get("status"),
                                           r.get("search", {}).get("expanded"))
                if r["step"] == 1 else (r["reason"] or "no collision -> no ALM"))]
        ax.set_title("\n".join(ttl), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = [], []
    for ax_ in axes[0]:
        for h, l in zip(*ax_.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("sample %d : prediction / ellipses / ALM correction / searched "
                 "path" % sample_id, fontsize=10, y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q64.yaml")
    ap.add_argument("--ckpt",
                    default="outputs/models/v5_c32q64/ckpt/final_best_ep138.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, nargs="+", default=[3009, 3011, 1001])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    plt = pyplot()
    ds = CarlaSplineDataset(args.split, cfg["data"]["processed_root"],
                            require_labels=False,
                            num_controls=int(cfg["model"]["num_controls"]),
                            num_safety_queries=int(
                                cfg["model"]["num_safety_queries"]))
    sid2pos = {int(s): i for i, s in enumerate(ds.sample_id)}
    sched = NoiseSchedule(16, "squaredcos_cap_v2").to(device)
    model = build_model(cfg).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    for sid in args.samples:
        if sid not in sid2pos:
            print("[skip] sample %d not in split %s" % (sid, args.split))
            continue
        batch, recs = replay(model, sched, ds, sid2pos[sid], cfg, device,
                             args.steps, seed=args.seed)
        last = recs[-1]
        final = last["safe"] if (last["alm_ok"] and last["safe"] is not None) \
            else last["raw"]
        hit_f, _ = collision_mask(batch["occupancy"][0], final)
        print("[replay] sample %d seed=%d  final coll=%.3f  cells(v/i)=%s  err=%s"
              "  (ALM: %s)" % (
                  sid, args.seed, float(hit_f.mean()),
                  [(r.get("cells_built", 0), r.get("cells_invalid", 0))
                   for r in recs[1:]], [r.get("cell_error") for r in recs[1:]],
                  ["%s/%.3f" % ("PASS" if r["alm_ok"] else
                                ("REJ" if r["alm_ran"] else "-"),
                                r["coll_before"]) for r in recs[1:]]))
        p = plot(plt, batch, recs, sid,
                 os.path.join(args.out, "stepdiag_%s_%d.png" % (args.split, sid)),
                 cfg=cfg)
        print("[fig]", p)


if __name__ == "__main__":
    main()

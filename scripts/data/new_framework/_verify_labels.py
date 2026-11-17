"""Final verification of every label array in the new root."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from src.grid_search import GridSpec, build_safe_grid  # noqa: E402

NEW = os.path.join(ROOT, "data", "processed", "160m-k4p-c32q64")
OLD = os.path.join(ROOT, "data", "processed", "160m-k4p-c48")
SC = 80.0
SPLITS = {"train": 3850, "val": 1550, "test": 420}
spec = GridSpec.from_resolution(256, 256, 256, scene_to_meter=SC)
summary, problems = {}, []

for split, n_expect in SPLITS.items():
    d, o = os.path.join(NEW, split), os.path.join(OLD, split)
    n_occ = int(np.load(os.path.join(d, "occupancy.npy"), mmap_mode="r").shape[0])
    if n_occ != n_expect:
        problems.append("%s occupancy rows %d != %d" % (split, n_occ, n_expect))
    row = {"n": n_occ}
    def chk(name, expect_rows):
        p = os.path.join(d, name + ".npy")
        if not os.path.exists(p):
            problems.append("%s/%s missing" % (split, name))
            return None
        a = np.load(p)
        if a.shape[0] != expect_rows:
            problems.append("%s/%s rows %d != %d" % (split, name, a.shape[0],
                                                     expect_rows))
        return a

    ctrl = chk("control_gt", n_occ)
    head = chk("heading", n_occ)
    hv = chk("heading_valid", n_occ)
    pp = chk("pdg_path_points", n_occ)
    pv = chk("pdg_path_valid", n_occ)
    ps = chk("pdg_path_status", n_occ)
    s4 = chk("ellipse_shape4_gt", n_occ)
    sv = chk("shape_valid", n_occ)
    cond = np.load(os.path.join(d, "conditions.npy"))

    if ctrl is not None:
        if ctrl.shape[1] != 32:
            problems.append("%s control_gt C=%d" % (split, ctrl.shape[1]))
        e = float(max(np.abs(ctrl[:, 0] - cond[:, 0]).max(),
                      np.abs(ctrl[:, -1] - cond[:, 1]).max()))
        row["control_endpoint_err"] = e
        if e > 1e-6:
            problems.append("%s control endpoints err %.2e" % (split, e))
        row["control_gt"] = dict(shape=list(ctrl.shape), min=float(ctrl.min()),
                                 max=float(ctrl.max()))
    if head is not None:
        nrm = np.linalg.norm(head, axis=-1)
        ok = hv.astype(bool)
        err = float(np.abs(nrm[ok] - 1.0).max()) if ok.any() else 0.0
        if err > 1e-5:
            problems.append("%s heading not unit (%.2e)" % (split, err))
        if not np.isfinite(head).all():
            problems.append("%s heading has NaN" % split)
        row["heading"] = dict(shape=list(head.shape), unit_err=err,
                              valid_frac=float(ok.mean()),
                              x_min=float(head[..., 0].min()),
                              x_max=float(head[..., 0].max()),
                              y_min=float(head[..., 1].min()),
                              y_max=float(head[..., 1].max()))
    if pp is not None:
        if not np.isfinite(pp).all():
            problems.append("%s pdg points have NaN" % split)
        ep = float(max(np.abs(pp[:, 0] - cond[:, 0]).max(),
                       np.abs(pp[:, -1] - cond[:, 1]).max()))
        if ep > 1e-6:
            problems.append("%s pdg endpoints not pinned (%.2e)" % (split, ep))
        seg = np.linalg.norm(np.diff(pp, axis=1), axis=2) * SC
        row["pdg_path_points"] = dict(shape=list(pp.shape), min=float(pp.min()),
                                      max=float(pp.max()),
                                      endpoint_err=ep,
                                      step_m_mean=float(seg.mean()),
                                      step_m_max=float(seg.max()),
                                      valid_frac=float(pv.mean()))
        vals, cnt = np.unique(ps, return_counts=True)
        row["pdg_path_status_counts"] = {int(k): int(v) for k, v in zip(vals, cnt)}
    if s4 is not None:
        if s4.shape[1] != 64:
            problems.append("%s ellipse Q=%d" % (split, s4.shape[1]))
        v = sv.astype(bool)
        ax = np.exp(s4[v][:, :2]) * SC if v.any() else None
        row["ellipse"] = dict(shape=list(s4.shape),
                              valid_frac=float(v.mean()),
                              samples_all_invalid=int((~v.any(axis=1)).sum()),
                              axis_m_mean=(None if ax is None else
                                           [float(ax[:, 0].mean()),
                                            float(ax[:, 1].mean())]),
                              axis_m_p95=(None if ax is None else
                                          [float(np.percentile(ax[:, 0], 95)),
                                           float(np.percentile(ax[:, 1], 95))]),
                              cos2t_range=[float(s4[..., 2].min()),
                                           float(s4[..., 2].max())],
                              sin2t_range=[float(s4[..., 3].min()),
                                           float(s4[..., 3].max())])
    # legacy comparison
    lo = os.path.join(o, "ellipse_shape4_gt.npy")
    if os.path.exists(lo):
        s4o = np.load(lo)
        vo = np.load(os.path.join(o, "shape_valid.npy")).astype(bool)
        if s4o.shape[1] == 128 and vo.any():
            a = np.exp(s4o[vo][:, :2]) * SC
            row["legacy128"] = dict(valid_frac=float(vo.mean()),
                                    axis_m_mean=[float(a[:, 0].mean()),
                                                 float(a[:, 1].mean())],
                                    axis_m_p95=[float(np.percentile(a[:, 0], 95)),
                                                float(np.percentile(a[:, 1], 95))])
    # geometry sanity on a few samples: the PDG path must lie on inflation-free cells
    bad = 0
    for i in range(0, n_occ, max(1, n_occ // 20)):
        occ = np.asarray(np.load(os.path.join(d, "occupancy.npy"),
                                 mmap_mode="r")[i])
        safe = build_safe_grid(occ, spec, robot_radius_m=0.8,
                               safety_margin_m=0.3, already_inflated_cells=0)
        rc = np.rint((np.asarray(pp[i], np.float64) + 1.0) / 2.0 * 256 - 0.5
                     ).astype(int).clip(0, 255)
        bad += int((~safe.free[rc[:, 1], rc[:, 0]]).sum())
    row["nonfree_path_points_sampled"] = bad
    if bad:
        problems.append("%s: %d path points outside the inflation-safe grid"
                        % (split, bad))
    summary[split] = row

print(json.dumps(summary, indent=2))
print("\nPROBLEMS:", problems if problems else "none")
with open(os.path.join(NEW, "_label_verification.json"), "w",
          encoding="utf-8") as fh:
    json.dump({"summary": summary, "problems": problems}, fh, indent=2)


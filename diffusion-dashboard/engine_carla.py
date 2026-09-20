"""CARLA + 32-control B-spline inference engine for the Diffusion Lens dashboard.

Same payload contract as the legacy Maze2D ``engine.py``, but:

  * the sample comes from the FIXED processed cache
    (``data/carla_processed/<split>/{conditions,curve_gt,occupancy,candidate_*}.npy``);
  * the skeleton graph + candidates are rebuilt ONLINE on the current occupancy
    (so the dashboard can still edit start/goal and draw obstacles);
  * the model is the control-space TrajSafe-Diffuser: the diffusion state is the
    32-control B-spline polygon Q_t, decoded to the 128-point curve with the
    fixed BSplineCodec; the ellipse centres are the FIXED Skeleton centres
    c_i = Gamma_m(i/127) (no learned progress);
  * the reverse trace records, for every timestep t=15..0, the noisy state that
    is fed to the model (``state_history``) and the model's x0 prediction
    (``x0_history``), plus the ellipse frame; ALM guidance is NOT migrated, so
    the ``alm`` channel only carries the coarse backbone and no regions.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

SITE_ROOT = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SITE_ROOT, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import (CandidateConfig, generate_candidates,
                                         normalized_dtw)
from src.models.trajsafe import TrajSafePlanner

CHECKPOINTS = {
    "best_task": os.path.join(ROOT, "outputs", "bspline_carla", "ckpt",
                              "best_task.pt"),
    "best": os.path.join(ROOT, "outputs", "bspline_carla", "ckpt", "best.pt"),
    "latest": os.path.join(ROOT, "outputs", "bspline_carla", "ckpt",
                           "latest.pt"),
    "best_run1": os.path.join(ROOT, "outputs", "bspline_carla", "ckpt",
                              "best_run1.pt"),
}


class Engine:
    def __init__(self, device, processed_root=None):
        self.device = device
        cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
        self.cfg = cfg
        self.processed_root = os.path.abspath(
            processed_root or cfg["data"].get("processed_root",
                                              "data/carla_processed"))
        self.geometry_points = int((cfg.get("topology") or {}).get(
            "candidate_geometry_points", 1280))
        self.cand_cfg = CandidateConfig.from_dict(cfg.get("topology") or {},
                                                  strict=False)
        self.skel_cfg = cfg.get("skeleton") or {}
        self.alm_cfg = cfg.get("alm") or {}
        self.schedule = NoiseSchedule(
            cfg["diffusion"]["timesteps"],
            beta_schedule=cfg["diffusion"].get("beta_schedule",
                                               "squaredcos_cap_v2"),
            beta_start=cfg["diffusion"].get("beta_start", 1e-4),
            beta_end=cfg["diffusion"].get("beta_end", 0.02)).to(device)
        self._models = {}
        self._splits = {}
        self._graphs = {}

    # ------------------------------------------------------------------ data
    def split_data(self, split):
        if split not in self._splits:
            d = os.path.join(self.processed_root, split)
            self._splits[split] = {
                "conditions": np.load(os.path.join(d, "conditions.npy")),
                "curve_gt": np.load(os.path.join(d, "curve_gt.npy")),
                "occupancy": np.load(os.path.join(d, "occupancy.npy"),
                                     mmap_mode="r"),
            }
        return self._splits[split]

    def sample_arrays(self, split, index):
        data = self.split_data(split)
        i = int(index)
        return (np.asarray(data["occupancy"][i], dtype=np.float32),
                np.asarray(data["conditions"][i], dtype=np.float32),
                np.asarray(data["curve_gt"][i], dtype=np.float32))

    # ----------------------------------------------------------------- model
    def get_model(self, model_id):
        if model_id not in CHECKPOINTS:
            raise ValueError("unknown checkpoint %r" % model_id)
        if model_id not in self._models:
            path = CHECKPOINTS[model_id]
            if not os.path.exists(path):
                raise FileNotFoundError("checkpoint not found: %s" % path)
            model = TrajSafePlanner(self.cfg["model"], self.cfg.get("ellipse_label"),
                                    self.cfg.get("bspline")).to(self.device)
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            model.eval()
            self._models[model_id] = (model, ckpt.get("epoch"))
        return self._models[model_id]

    @staticmethod
    def occupancy_key(occupancy):
        """The skeleton graph depends ONLY on the occupancy, so the cache key is
        a hash of the occupancy itself: editing start/goal reuses the graph, but
        adding/removing an obstacle rebuilds it (no stale routes)."""
        import hashlib
        arr = np.ascontiguousarray(np.asarray(occupancy) > 0.5)
        return hashlib.md5(arr.tobytes()).hexdigest()[:10]

    def get_graph(self, key, occupancy):
        if key not in self._graphs:
            self._graphs[key] = build_skeleton_graph(
                occupancy,
                safety_dilation_cells=int(self.skel_cfg.get(
                    "safety_dilation_cells", 1)),
                thinning_backend=str(self.skel_cfg.get("thinning_backend",
                                                       "auto")),
                pure_cycle_aux_nodes=int(self.skel_cfg.get(
                    "pure_cycle_aux_nodes", 2)))
            if len(self._graphs) > 32:
                self._graphs.pop(next(iter(self._graphs)))
        return self._graphs[key]

    # ------------------------------------------------------------- packaging
    @staticmethod
    def _rounded(array):
        if torch.is_tensor(array):
            array = array.detach().cpu()
        return np.round(np.asarray(array, dtype=np.float64), 5).tolist()

    def _pack_candidates(self, cands, condition):
        M = int(cands.num_slots)
        L = int(self.cand_cfg.candidate_points)
        xy = np.asarray(cands.coords, dtype=np.float32).reshape(1, M, L, 2)
        mask = np.asarray(cands.mask, dtype=bool).reshape(1, M)
        lengths = np.zeros(M, dtype=np.int64)
        dense = []
        for m in range(M):
            if bool(cands.mask[m]) and m < len(cands.geometry):
                geom = np.asarray(cands.geometry[m], dtype=np.float32).reshape(-1, 2)
                lengths[m] = len(geom)
                dense.append(geom)
            else:
                dense.append(np.zeros((0, 2), dtype=np.float32))
        G = max(int(self.geometry_points), int(lengths.max()) if M else 1)
        geometry = np.zeros((1, M, G, 2), dtype=np.float32)
        start = np.asarray(condition[0], dtype=np.float32)
        goal = np.asarray(condition[1], dtype=np.float32)
        for m in range(M):
            n = int(lengths[m])
            if n <= 0:
                continue
            geometry[0, m, :n] = dense[m]
            if n >= 2:
                geometry[0, m, 0] = start
                geometry[0, m, n - 1] = goal
        return {
            "xy": torch.as_tensor(xy, device=self.device),
            "mask": torch.as_tensor(mask, device=self.device),
            "geometry": torch.as_tensor(geometry, device=self.device),
            "geometry_lengths": torch.as_tensor(lengths[None],
                                                device=self.device),
            "geometry_points": int(G),
        }

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def _run_ddim(self, model, cond, occ, packed, seed):
        torch.manual_seed(int(seed))
        sched = self.schedule
        T = int(sched.num_timesteps)
        dev = self.device
        q = torch.randn(1, int(model.num_controls), 2, device=dev)
        q[:, 0] = cond[:, 0]
        q[:, -1] = cond[:, 1]
        sqrt_ab = sched.sqrt_alphas_cumprod.detach().to(dev).float()
        sqrt_1ma = sched.sqrt_one_minus_alphas_cumprod.detach().to(dev).float()
        steps = []
        last = None
        for t in range(T - 1, -1, -1):
            q = model.hard_control_endpoints(q, cond)
            tb = torch.full((1,), t, device=dev, dtype=torch.long)
            ab = sqrt_ab[t].expand(1).contiguous()
            out = model.forward_all(q, occ, cond, tb, ab, packed["xy"],
                                    packed["mask"], packed["geometry"],
                                    packed["geometry_lengths"], select_index=None)
            last = out
            steps.append({
                "t": int(t),
                "state": out["input_curve"][0],           # decoded noisy state
                "x0": out["final"][0],                    # model x0 prediction
                "coarse": out["coarse"][0],
                "control": out["control"][0],
                "selected_idx": int(out["selected_idx"][0]),
                "pi": out["topo"]["pi"][0],
                "ellipse": out["ellipse"],
            })
            q0 = out["control"]
            if t > 0:
                eps = (q - sqrt_ab[t] * q0) / sqrt_1ma[t]
                q = sqrt_ab[t - 1] * q0 + sqrt_1ma[t - 1] * eps
            else:
                q = q0
        # close the replay with the final x0 frame
        steps.append({"t": -1, "state": steps[-1]["x0"], "x0": steps[-1]["x0"],
                      "coarse": steps[-1]["coarse"],
                      "control": steps[-1]["control"],
                      "selected_idx": steps[-1]["selected_idx"],
                      "pi": steps[-1]["pi"], "ellipse": steps[-1]["ellipse"]})
        return steps, last

    # ------------------------------------------------------------- metrics
    @staticmethod
    def _free_mask(occupancy, points):
        res = occupancy.shape[0]
        px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
        py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
        ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
        free = np.zeros(len(points), dtype=bool)
        free[ok] = ~occupancy[py[ok], px[ok]].astype(bool)
        return free

    def _ellipse_metrics(self, occupancy, steps):
        hit, total = 0, 0
        ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ca, sa = np.cos(ang)[None], np.sin(ang)[None]
        for st in steps:
            ell = st["ellipse"]
            center = ell["center"][0].detach().cpu().numpy()
            a = ell["a"][0].detach().cpu().numpy()
            b = ell["b"][0].detach().cpu().numpy()
            th = ell["theta"][0].detach().cpu().numpy()
            ct, stt = np.cos(th)[:, None], np.sin(th)[:, None]
            ex, ey = a[:, None] * ca, b[:, None] * sa
            ring = np.stack([ct * ex - stt * ey + center[:, 0:1],
                             stt * ex + ct * ey + center[:, 1:2]], axis=-1)
            for k in range(len(center)):
                if not self._free_mask(occupancy, ring[k]).all():
                    hit += 1
            total += len(center)
        last_center = steps[-1]["ellipse"]["center"][0].detach().cpu().numpy()
        res = occupancy.shape[0]
        j, i = np.nonzero(occupancy.astype(bool))
        out = {"ellipse_count": int(total),
               "ellipse_collision_rate": float(hit / total) if total else None,
               "center_free_rate": float(self._free_mask(
                   occupancy, last_center).mean()),
               "min_center_clearance_cells": None}
        if len(j):
            cx = (i + 0.5) * 2.0 / res - 1.0
            cy = (j + 0.5) * 2.0 / res - 1.0
            d = np.sqrt((last_center[:, 0:1] - cx[None, :]) ** 2
                        + (last_center[:, 1:2] - cy[None, :]) ** 2)
            out["min_center_clearance_cells"] = float(d.min() * res / 2.0)
        return out

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, sample_key, split, index, occupancy, condition, seed,
                 model_id="best_task", verify_regions=False):
        model, epoch = self.get_model(model_id)
        condition = np.asarray(condition, dtype=np.float32).reshape(2, 2)
        occupancy = np.asarray(occupancy, dtype=np.float32)
        graph = self.get_graph(
            "%s:%d:%s" % (split, int(index), self.occupancy_key(occupancy)),
            occupancy)
        cands = generate_candidates(graph, condition[0], condition[1],
                                    self.cand_cfg)
        if cands.num_valid == 0:
            raise ValueError("该起终点在当前 occupancy 上没有合法候选路径"
                             "（起终点可能落在障碍里，或自定义障碍截断了通路）。")
        packed = self._pack_candidates(cands, condition)
        cond = torch.as_tensor(condition[None], dtype=torch.float32,
                               device=self.device)
        occ = torch.as_tensor(occupancy, dtype=torch.float32,
                              device=self.device)[None, None]
        steps, _ = self._run_ddim(model, cond, occ, packed, seed)

        state_history, x0_history, ellipse_history = [], [], []
        control_history = []
        alm_frames, topology_frames = [], []
        for st in steps:
            ell = st["ellipse"]
            center = ell["center"][0].detach().cpu().numpy()
            shape4 = ell["shape4"][0].detach().cpu().numpy()
            state_history.append(self._rounded(st["state"]))
            x0_history.append(self._rounded(st["x0"]))
            # the 32-control B-spline polygon the network actually predicts
            control_history.append(self._rounded(st["control"]))
            ellipse_history.append({"center": self._rounded(center),
                                    "shape4": self._rounded(shape4)})
            # ALM is not migrated: the channel only carries the coarse backbone
            alm_frames.append({"t": int(st["t"]),
                               "raw_p": self._rounded(st["coarse"]),
                               "enforced": [], "regions": [], "stats": {}})
            topology_frames.append({
                "t": int(st["t"]),
                "selected_idx": int(st["selected_idx"]),
                "pi": self._rounded(st["pi"]),
                "progress": self._rounded(
                    model.fixed_progress[None].expand(1, model.horizon)[0]),
                "center": self._rounded(center),
                "shape4": self._rounded(shape4),
                "regions": [],
            })
        labels = ["t=%d" % int(st["t"]) for st in steps[:-1]] + ["x0"]

        selected = int(steps[-1]["selected_idx"])
        final_curve = np.asarray(x0_history[-1], dtype=np.float64)
        metrics = self._ellipse_metrics(occupancy, steps)
        metrics["traj_collision"] = bool(
            not self._free_mask(occupancy, final_curve).all())
        metrics["region_count"] = 0
        metrics["progress_monotonic"] = True
        selections = [int(f["selected_idx"]) for f in topology_frames]
        metrics["step_jitter"] = int(sum(1 for a, b in zip(selections[:-1],
                                                           selections[1:])
                                         if a != b))
        metrics["selection_changes"] = int(len(set(selections)) > 1)

        best = None
        if cands.num_valid:
            d = [normalized_dtw(final_curve, cands.metric_polyline(int(m)))
                 for m in cands.valid_index()]
            best = int(cands.valid_index()[int(np.argmin(d))])
        metrics["best_candidate"] = best
        metrics["curve_rmse_m"] = None
        metrics["selected_ndtw"] = float(normalized_dtw(
            final_curve, cands.metric_polyline(selected)))

        return {
            "sample_key": sample_key, "model_id": model_id, "seed": int(seed),
            "cache_hit": False, "condition": condition.tolist(),
            "state_labels": labels,
            "schedule": {
                "sqrt_alpha_bar": np.round(
                    self.schedule.sqrt_alphas_cumprod.cpu().numpy(), 8).tolist(),
                "sqrt_one_minus_alpha_bar": np.round(
                    self.schedule.sqrt_one_minus_alphas_cumprod.cpu().numpy(),
                    8).tolist(),
            },
            "state_history": state_history,
            "x0_history": x0_history,
            # 32-control polygons: [T+1, 32, 2] in scene coords.  These are the
            # diffusion state q0_hat; the drawn curves are their B-spline decode.
            "control_history": control_history,
            "ellipse_history": ellipse_history,
            "alm": {"enabled": False, "start_t": 0, "frames": alm_frames},
            "topology": {
                "epoch": epoch,
                "selection": "argmax",
                "num_candidates": int(cands.num_slots),
                "num_valid": int(cands.num_valid),
                "raw_k": int(cands.raw_k),
                "selected_idx": selected,
                "pi": self._rounded(steps[-1]["pi"]),
                "topology_path": self._rounded(cands.coords[selected]),
                "candidate_paths": [self._rounded(cands.coords[m])
                                    for m in range(cands.num_slots)],
                "candidate_mask": [bool(v) for v in cands.mask.tolist()],
                "candidate_lengths": self._rounded(cands.lengths),
                "geometry_points": int(packed["geometry_points"]),
                "node_path": [int(v) for v in cands.node_paths[selected]]
                if cands.node_paths else [],
                "frames": topology_frames,
                "metrics": metrics,
                "skeleton": {"nodes": len(graph.nodes),
                             "branches": len(graph.branches)},
            },
        }

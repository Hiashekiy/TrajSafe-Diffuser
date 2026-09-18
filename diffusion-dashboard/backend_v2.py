"""V2 engine for the diffusion dashboard.

Runs the Skeleton-Topology-Grounded Trajectory Diffusion model
(ONE trajectory diffusion, c_i = gamma_m(s_i)) and returns a payload with the
same shape as the V1 backend so the existing replay UI keeps working:

    PHistory / X0PHistory      [steps+1][128][2]     trajectory states
    E6History / X0E6History    [steps+1][128][6]     ellipses, V1 6-vector form
                                                     [dx, dy, log a, log b,
                                                      cos 2t, sin 2t]
                                                     with dx,dy relative to the
                                                     matching trajectory point,
                                                     so the renderer needs no
                                                     change at all
    alm.frames[i]              convex regions (verified) + the un-conditioned
                               x0 prediction, reusing the ALM overlay channel
    v2                         V2 specific diagnostics (commit step, selected
                               candidate, pi, safety metrics)

Only the ellipse CENTRE differs semantically from V1: it is gamma_m(s_i), a
point of the committed topology, never p_i + delta_c_i.

Self test (no HTTP):
    python backend_v2.py --selftest --sample umaze-9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

SITE_ROOT = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SITE_ROOT, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.diffusion.sampler_v2 import sample_v2
from src.diffusion.schedule import NoiseSchedule
from src.geometry.safe_convex_region import generate_verified_convex_region
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import CandidateConfig, generate_candidates
from src.models.skeleton import SkeletonPlanner
from src.utils.config import load_config

CONFIG_PATH = os.path.join(ROOT, "configs", "config_v2_skeleton.yaml")
MAZES = ("umaze", "medium", "large")
V2_CHECKPOINTS = {
    "v2_best": "outputs/ckpt_v2_skeleton/best.pt",
    "v2_latest": "outputs/ckpt_v2_skeleton/latest.pt",
}
STATE_LABELS = ["t=%d" % t for t in range(15, -1, -1)] + ["x0"]
DEFAULT_TINY_ELLIPSE = 0.02          # scene units, used before the commit step


class V2Engine:
    def __init__(self, device):
        self.cfg = load_config(CONFIG_PATH)
        self.topo_cfg = dict(self.cfg.get("topology") or {})
        self.cand_cfg = CandidateConfig.from_dict(self.topo_cfg, strict=False)
        self.commit_t = int(self.topo_cfg.get("commit_t", 7))
        self.selection = str(self.topo_cfg.get("selection", "sample"))
        self.skeleton_cfg = dict(self.cfg.get("skeleton") or {})
        self.device = device
        self.maps = {name: np.load(os.path.join(ROOT, "data", "processed_scene_v1",
                                                "maps", "%s.npy" % name))
                     for name in MAZES}
        self.schedule = NoiseSchedule(
            self.cfg["diffusion"]["timesteps"],
            beta_schedule=self.cfg["diffusion"].get("beta_schedule",
                                                    "squaredcos_cap_v2"),
            beta_start=self.cfg["diffusion"].get("beta_start", 0.0001),
            beta_end=self.cfg["diffusion"].get("beta_end", 0.02)).to(device)
        self.models: dict = {}
        self.graphs: dict = {}

    # ------------------------------------------------------------------ model
    def get_model(self, model_id):
        if model_id not in self.models:
            path = os.path.join(ROOT, V2_CHECKPOINTS[model_id])
            model = SkeletonPlanner(self.cfg["model"], self.topo_cfg).to(self.device)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            model.eval()
            self.models[model_id] = (model, ckpt.get("epoch"))
        return self.models[model_id]

    # ------------------------------------------------------------------ graph
    def get_graph(self, maze, occupancy):
        key = (maze, hashlib.sha1(
            np.ascontiguousarray((occupancy > 0.5).astype(np.uint8)).tobytes()
        ).hexdigest()[:16])
        if key not in self.graphs:
            self.graphs[key] = build_skeleton_graph(
                occupancy,
                safety_dilation_cells=int(
                    self.skeleton_cfg.get("safety_dilation_cells", 1)),
                thinning_backend=str(
                    self.skeleton_cfg.get("thinning_backend", "auto")),
                pure_cycle_aux_nodes=int(
                    self.skeleton_cfg.get("pure_cycle_aux_nodes", 2)))
            if len(self.graphs) > 12:                 # keep the cache bounded
                self.graphs.pop(next(iter(self.graphs)))
        return self.graphs[key]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _rounded(array):
        if torch.is_tensor(array):
            array = array.detach().cpu()
        return np.round(np.asarray(array, dtype=np.float64), 5).tolist()

    @staticmethod
    def _e6(center, shape4, trajectory):
        """(centre, shape4, path point) -> V1 six-vector [dx,dy,loga,logb,c2,s2]."""
        off = np.asarray(center, dtype=np.float64) - np.asarray(trajectory,
                                                                dtype=np.float64)
        return np.concatenate([off, np.asarray(shape4, dtype=np.float64)], axis=-1)

    def _tiny_e6(self, trajectory):
        r0 = float(np.log(DEFAULT_TINY_ELLIPSE))
        zeros = np.zeros_like(np.asarray(trajectory, dtype=np.float64))
        shape = np.tile(np.array([r0, r0, 1.0, 0.0]), (len(zeros), 1))
        return np.concatenate([zeros, shape], axis=-1)

    def _regions(self, occupancy, center, shape4, stride=8):
        out = []
        for j in range(0, len(center), max(1, int(stride))):
            a = float(np.exp(shape4[j, 0]))
            b = float(np.exp(shape4[j, 1]))
            theta = float(0.5 * np.arctan2(shape4[j, 3], shape4[j, 2]))
            try:
                _, _, verts, info = generate_verified_convex_region(
                    occupancy, center[j], a, b, theta, window_half=0.5,
                    safety_margin=0.008, shrink=0.002)
            except Exception:
                continue
            if verts is None or len(verts) < 3 or not info.get("safe", False):
                continue
            out.append({"i": int(j), "polygon": self._rounded(verts)})
        return out

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, sample_key, dataset_id, maze, occupancy, condition, seed,
                 model_id="v2_best", verify_regions=True):
        model, epoch = self.get_model(model_id)
        start = np.asarray(condition[0], dtype=np.float64)
        goal = np.asarray(condition[1], dtype=np.float64)

        graph = self.get_graph(maze, occupancy)
        cands = generate_candidates(graph, start, goal, self.cand_cfg)
        if cands.num_valid == 0:
            raise ValueError(
                "该起终点在 V2 骨架图上没有合法候选路径（起终点可能落在障碍里，"
                "或添加的自定义障碍把通路截断了）。")

        cond = torch.as_tensor(np.asarray(condition, dtype=np.float32)[None],
                               device=self.device)
        occ = torch.as_tensor(occupancy, dtype=torch.float32,
                              device=self.device)[None, None]
        cand = torch.as_tensor(cands.paths[None], device=self.device)
        vmask = torch.as_tensor(cands.mask[None], device=self.device)
        clen = torch.as_tensor(cands.lengths[None], device=self.device)

        result = sample_v2(model, self.schedule, cond, occ, cand, vmask, clen,
                           device=self.device, seed=seed, commit_t=self.commit_t,
                           selection=self.selection, return_trace=True)
        trace = result["trace"]

        p_history, e6_history, x0p_history, x0e6_history = [], [], [], []
        alm_frames = []
        v2_frames = []
        for step in trace:
            p = step["p"][0].numpy()
            x0 = step["x0_p"][0].numpy()
            base = step["x0_p_base"][0].numpy()
            center = (step["ellipse_center"][0].numpy()
                      if step["ellipse_center"] is not None else None)
            shape4 = (step["ellipse_shape4"][0].numpy()
                      if step["ellipse_shape4"] is not None else None)
            p_history.append(self._rounded(p))
            x0p_history.append(self._rounded(x0))
            if center is None:
                e6_history.append(self._rounded(self._tiny_e6(p)))
                x0e6_history.append(self._rounded(self._tiny_e6(x0)))
                alm_frames.append(None)
                v2_frames.append({"t": step["t"], "committed": False,
                                  "selectedIdx": int(step["selected_idx"][0]),
                                  "pi": self._rounded(step["pi"][0]),
                                  "regions": [], "progress": None})
            else:
                e6_history.append(self._rounded(self._e6(center, shape4, p)))
                x0e6_history.append(self._rounded(self._e6(center, shape4, x0)))
                regions = (self._regions(occupancy, center, shape4)
                           if verify_regions else [])
                alm_frames.append({"t": step["t"], "rawP": self._rounded(base),
                                   "enforced": [], "regions": regions,
                                   "stats": {}})
                v2_frames.append({
                    "t": step["t"], "committed": True,
                    "selectedIdx": int(step["selected_idx"][0]),
                    "pi": self._rounded(step["pi"][0]),
                    "regions": regions,
                    "progress": self._rounded(step["progress"][0]),
                })
        p_history.append(p_history[-1])
        e6_history.append(e6_history[-1])
        x0p_history.append(x0p_history[-1])
        x0e6_history.append(x0e6_history[-1])
        alm_frames.append(None)
        v2_frames.append(dict(v2_frames[-1]))

        selected = int(result["selected_idx"][0])
        topology = cands.coords[selected] if cands.num_valid else np.zeros((128, 2))
        metrics = self._metrics(occupancy, p_history[-1], p_history, x0_history=None,
                                center_all=None)
        metrics.update(self._ellipse_metrics(occupancy, trace))
        metrics["regionCount"] = int(sum(len(f["regions"]) for f in v2_frames
                                         if f and f.get("regions")))
        metrics["progressMonotonic"] = all(
            bool(np.all(np.diff(f["progress"]) >= -1e-6))
            for f in v2_frames if f and f.get("progress"))

        return {
            "sampleKey": sample_key, "modelId": model_id, "seed": seed,
            "cacheHit": False, "condition": np.asarray(condition).tolist(),
            "obstacles": [],
            "stateLabels": STATE_LABELS,
            "schedule": {
                "sqrtAlphaBar": np.round(
                    self.schedule.sqrt_alphas_cumprod.cpu().numpy(), 8).tolist(),
                "sqrtOneMinusAlphaBar": np.round(
                    self.schedule.sqrt_one_minus_alphas_cumprod.cpu().numpy(),
                    8).tolist(),
            },
            "PHistory": p_history, "E6History": e6_history,
            "X0PHistory": x0p_history, "X0E6History": x0e6_history,
            # reused overlay channel: regions + the un-conditioned x0 prediction
            "alm": {"enabled": False, "startT": self.commit_t, "frames": alm_frames},
            "v2": {
                "engine": "skeleton-topology-grounded",
                "epoch": epoch,
                "commitT": self.commit_t,
                "selection": self.selection,
                "numCandidates": int(cands.num_slots),
                "numValid": int(cands.num_valid),
                "rawK": int(cands.raw_k),
                "selectedIdx": selected,
                "pi": self._rounded(result["topology_pi"][0]),
                "topologyPath": self._rounded(topology),
                "candidateLengths": self._rounded(cands.lengths),
                "nodePath": [int(v) for v in cands.node_paths[selected]]
                if cands.node_paths else [],
                "frames": v2_frames,
                "metrics": metrics,
                "skeleton": {"nodes": len(graph.nodes),
                             "branches": len(graph.branches)},
            },
        }

    # --------------------------------------------------------------- metrics
    @staticmethod
    def _metrics(occupancy, trajectory, p_history, x0_history, center_all):
        res = occupancy.shape[0]

        def collides(points):
            px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
            py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
            inside = (px >= 0) & (px < res) & (py >= 0) & (py < res)
            if not inside.all():
                return True
            return bool(occupancy[py, px].astype(bool).any())

        return {"trajCollision": bool(collides(np.asarray(trajectory)))}

    @staticmethod
    def _ellipse_metrics(occupancy, trace):
        """CenterFree + per-point ellipse collision over the refined steps."""
        res = occupancy.shape[0]

        def free(points):
            px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
            py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
            ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
            out = np.zeros(len(points), dtype=bool)
            out[ok] = ~occupancy[py[ok], px[ok]].astype(bool)
            return out

        centers, hit, total = [], 0, 0
        last = None
        for step in trace:
            if step["ellipse_center"] is None:
                continue
            last = step
            center = step["ellipse_center"][0].numpy()
            shape4 = step["ellipse_shape4"][0].numpy()
            centers.append(center)
            a = np.exp(shape4[:, 0])
            b = np.exp(shape4[:, 1])
            th = 0.5 * np.arctan2(shape4[:, 3], shape4[:, 2])
            ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
            ex = a[:, None] * np.cos(ang)[None]
            ey = b[:, None] * np.sin(ang)[None]
            ring = np.stack([ct * ex - st * ey + center[:, 0:1],
                             st * ex + ct * ey + center[:, 1:2]], axis=-1)
            for k in range(len(center)):
                if not free(ring[k]).all():
                    hit += 1
            total += len(center)
        out = {"ellipseCount": int(total),
               "ellipseCollisionRate": float(hit / total) if total else None,
               "centerFreeRate": None, "minCenterClearanceCells": None}
        if centers:
            center = centers[-1]
            out["centerFreeRate"] = float(free(center).mean())
            j, i = np.nonzero(occupancy.astype(bool))
            cx = (i + 0.5) * 2.0 / res - 1.0
            cy = (j + 0.5) * 2.0 / res - 1.0
            d = np.sqrt((center[:, 0:1] - cx[None, :]) ** 2
                        + (center[:, 1:2] - cy[None, :]) ** 2)
            out["minCenterClearanceCells"] = float(d.min() * res / 2.0)
        return out


def _selftest(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    engine = V2Engine(device)
    with open(os.path.join(SITE_ROOT, "lib", "dashboard-catalog.json"),
              "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    sample = next(s for s in catalog["samples"] if s["key"] == args.sample)
    occupancy = engine.maps[sample["maze"]]
    payload = engine.generate(sample["key"], int(sample["datasetId"]),
                              sample["maze"], occupancy, sample["condition"],
                              args.seed, args.model)
    v2 = payload["v2"]
    print(json.dumps({"model": args.model, "epoch": v2["epoch"],
                      "maze": sample["maze"], "sample": sample["key"],
                      "commitT": v2["commitT"], "numValid": v2["numValid"],
                      "selectedIdx": v2["selectedIdx"], "pi": v2["pi"],
                      "metrics": v2["metrics"], "skeleton": v2["skeleton"],
                      "steps": len(v2["frames"])}, indent=2))
    shapes = {k: np.asarray(payload[k]).shape for k in
              ("PHistory", "E6History", "X0PHistory", "X0E6History")}
    print("payload shapes:", shapes)
    print("E6 last[0]:", payload["E6History"][-1][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample", default="umaze-9")
    ap.add_argument("--model", default="v2_best")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--device", default=None)
    _selftest(ap.parse_args())

"""V3 engine for the diffusion dashboard.

This is the report-faithful TrajSafe-Diffuser:

    P_t -> H_traj -> {R_m} -> m = argmax(pi) -> H_prog -> s
        -> c = Gamma_m(s) -> H_ell -> H_clean -> P0_hat -> DDIM

Key dashboard-facing differences from V2:

  * every reverse timestep re-scores the topology (no commit timestep);
  * the candidate set is generated ONLINE from the current occupancy and
    start/goal with the SAME generator as the offline preprocessing, so the
    user-drawn obstacles and edited endpoints are part of the search graph;
  * all candidate search paths are returned in ``v2.candidatePaths`` (the
    existing V2 UI block is reused, with ``engine='v3-skeleton-dynamic'``);
  * each step returns the selected candidate, pi, progress and its ellipse
    centre/shape so the replay shows whether the topology switches.

Self test (no HTTP):

    python backend_v3.py --selftest --sample large-855
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

from src.diffusion.alm_guidance import alm_correct
from src.diffusion.sampler_v3 import sample_v3
from src.diffusion.schedule import NoiseSchedule
from src.geometry.convex_corridor import EllipseRegionBuilder
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import CandidateConfig, generate_candidates
from src.models.skeleton_v3 import SkeletonPlannerV3
from src.utils.config import load_config

CONFIG_PATH = os.path.join(ROOT, "configs", "config_v3_skeleton.yaml")
ALM_CONFIG_PATH = os.path.join(ROOT, "configs", "config_v3_alm.yaml")
MAZES = ("umaze", "medium", "large")
V3_CHECKPOINTS = {
    "v3_best": "outputs/ckpt_v3_skeleton/best.pt",
    "v3_latest": "outputs/ckpt_v3_skeleton/latest.pt",
}
DEFAULT_GEOMETRY_POINTS = 1280


def polygon_vertices(A, b, mask, tol=1e-5):
    """Ordered vertices of a bounded 2-D halfspace intersection (scene frame)."""
    A, b = np.asarray(A)[mask], np.asarray(b)[mask]
    candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            matrix = np.stack((A[i], A[j]))
            det = np.linalg.det(matrix)
            if abs(det) <= 1e-8:
                continue
            point = np.linalg.solve(matrix, np.asarray((b[i], b[j])))
            if np.all(A @ point <= b + tol):
                candidates.append(point)
    if len(candidates) < 3:
        return []
    vertices = np.unique(np.round(np.asarray(candidates), 7), axis=0)
    center = vertices.mean(axis=0)
    angle = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return np.round(vertices[np.argsort(angle)], 5).tolist()


class V3Engine:
    def __init__(self, device):
        self.cfg = load_config(CONFIG_PATH)
        self.topo_cfg = dict(self.cfg.get("topology") or {})
        self.cand_cfg = CandidateConfig.from_dict(self.topo_cfg, strict=False)
        self.skeleton_cfg = dict(self.cfg.get("skeleton") or {})
        self.geometry_points = int(
            self.topo_cfg.get("candidate_geometry_points", DEFAULT_GEOMETRY_POINTS))
        self.alm_cfg = (load_config(ALM_CONFIG_PATH).get("alm") or {})
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
            path = os.path.join(ROOT, V3_CHECKPOINTS[model_id])
            model_cfg = dict(self.cfg["model"])
            model_cfg["assert_shapes"] = False          # inference speed
            model = SkeletonPlannerV3(model_cfg,
                                      self.cfg.get("ellipse_label")).to(self.device)
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
            if len(self.graphs) > 8:                    # keep the cache bounded
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

    def _pack_candidates(self, cands, condition):
        """CandidateSet -> V3 tensors [1,M,...] on device.

        Candidate geometry is the dense safe Skeleton Curve Gamma_m; it is padded
        to a common length and its exact start/goal endpoints are restored, the
        same convention as ``SkeletonDatasetV3``.
        """
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
            "geometry_lengths": torch.as_tensor(lengths[None], device=self.device),
            "geometry_points": int(G),
        }

    # ---------------------------------------------------------- ALM guidance
    def _make_guidance(self, occupancy):
        """Inference-time V3 ALM guidance: predicted ellipse -> verified region.

        The correction is applied to the model's raw ``P0_hat`` before DDIM; the
        network weights, training losses and the report architecture are
        unchanged.
        """
        cfg = dict(self.alm_cfg)
        start_t = int(cfg.get("start_t", 7))
        builder = EllipseRegionBuilder(
            torch.as_tensor(occupancy, dtype=torch.float32,
                            device=self.device)[None, None], cfg)
        inner_steps = int(cfg.get("inner_steps", 4))
        rho = float(cfg.get("rho", 5.0))
        step_size = float(cfg.get("step_size", 0.03))
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        max_correction = float(cfg.get("max_correction_per_step", 0.10))
        proximity = float(cfg.get("proximity_weight", 1.0))
        smooth = float(cfg.get("correction_smooth_weight", 20.0))

        def guide(x0, out, p_t, t):
            if int(t) > start_t:
                return x0, None
            ell = out["ellipse"]
            shape4 = ell["shape4"]
            e6 = torch.cat([ell["center"] - x0, shape4], dim=-1)   # [B,H,6]
            A, b, face_mask, valid = builder(x0, e6)
            enforce = builder.segment_needs_guidance(x0)
            lam = torch.zeros(x0.shape[0], x0.shape[1] - 1,
                              device=x0.device, dtype=x0.dtype)
            corrected, _, stats = alm_correct(
                x0, A[:, 1:], b[:, 1:], face_mask[:, 1:], valid[:, 1:],
                lam, rho, step_size=step_size, inner_steps=inner_steps,
                max_grad_norm=max_grad_norm,
                max_correction_per_step=max_correction,
                proximity_weight=proximity,
                correction_smooth_weight=smooth,
                enforce_mask=enforce, collect_stats=True)

            horizon = x0.shape[1]
            cpu_A, cpu_b, cpu_mask = (A[0].cpu().numpy(), b[0].cpu().numpy(),
                                      face_mask[0].cpu().numpy())
            enforce_cpu = enforce[0].cpu().numpy()
            region_indices = set(range(8, horizon, 8)) | {horizon - 1}
            if int(enforce_cpu.sum()) <= 40:
                for s in range(horizon - 1):
                    if bool(enforce_cpu[s]):
                        region_indices.add(int(s) + 1)
            regions = []
            for j in sorted(region_indices):
                if not bool(valid[0, j].cpu()):
                    continue
                polygon = polygon_vertices(cpu_A[j], cpu_b[j], cpu_mask[j])
                if polygon:
                    regions.append({"i": int(j), "polygon": polygon})
            info = {
                "regions": regions,
                "enforced": [int(s) for s in range(horizon - 1)
                             if bool(enforce_cpu[s])],
                "rawP": self._rounded(x0[0]),
                "stats": {k: float(v.detach().cpu()) for k, v in stats.items()},
            }
            return corrected, info

        return guide


    @staticmethod
    def _traj_collision(occupancy, trajectory):
        res = occupancy.shape[0]
        px = np.rint((trajectory[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
        py = np.rint((trajectory[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
        inside = (px >= 0) & (px < res) & (py >= 0) & (py < res)
        if not inside.all():
            return True
        return bool(occupancy[py, px].astype(bool).any())

    @staticmethod
    def _ellipse_metrics(occupancy, trace):
        """Collision / center-free / clearance over the whole reverse trace."""
        res = occupancy.shape[0]

        def free(points):
            px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
            py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
            ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
            out = np.zeros(len(points), dtype=bool)
            out[ok] = ~occupancy[py[ok], px[ok]].astype(bool)
            return out

        hit, total = 0, 0
        centers = []
        for step in trace:
            center = step["ellipse_center"][0].numpy()
            a = step["ellipse_a"][0].numpy()
            b = step["ellipse_b"][0].numpy()
            th = step["ellipse_theta"][0].numpy()
            centers.append(center)
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

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, sample_key, dataset_id, maze, occupancy, condition,
                 seed, model_id="v3_best", verify_regions=False):
        model, epoch = self.get_model(model_id)
        condition = np.asarray(condition, dtype=np.float32).reshape(2, 2)
        start = condition[0]
        goal = condition[1]

        graph = self.get_graph(maze, occupancy)
        cands = generate_candidates(graph, start, goal, self.cand_cfg)
        if cands.num_valid == 0:
            raise ValueError(
                "该起终点在骨架图上没有合法候选路径（起终点可能落在障碍里，"
                "或添加的自定义障碍把通路截断了）。")

        packed = self._pack_candidates(cands, condition)
        cond = torch.as_tensor(condition[None], dtype=torch.float32,
                               device=self.device)
        occ = torch.as_tensor(occupancy, dtype=torch.float32,
                              device=self.device)[None, None]
        guide = self._make_guidance(occupancy) if verify_regions else None
        result = sample_v3(model, self.schedule, cond, occ,
                           packed["xy"], packed["mask"], packed["geometry"],
                           packed["geometry_lengths"], device=self.device,
                           seed=seed, return_trace=True, alm_guidance=guide)
        trace = result["trace"]

        p_history, e6_history, x0p_history, x0e6_history = [], [], [], []
        alm_frames, v3_frames = [], []
        region_count = 0
        for step in trace:
            p = step["p"][0].numpy()
            x0 = step["final"][0].numpy()
            coarse = step["coarse"][0].numpy()
            center = step["ellipse_center"][0].numpy()
            shape4 = step["ellipse_shape4"][0].numpy()
            p_history.append(self._rounded(p))
            x0p_history.append(self._rounded(x0))
            e6_history.append(self._rounded(self._e6(center, shape4, p)))
            x0e6_history.append(self._rounded(self._e6(center, shape4, x0)))
            guide_info = step.get("guide")
            if guide is None:
                # report-faithful run: reuse the overlay for the coarse branch
                alm_frames.append({"t": int(step["t"]),
                                   "rawP": self._rounded(coarse),
                                   "enforced": [], "regions": [], "stats": {}})
            elif guide_info is not None:
                region_count += len(guide_info["regions"])
                alm_frames.append({
                    "t": int(step["t"]),
                    "rawP": guide_info["rawP"],
                    "enforced": guide_info["enforced"],
                    "regions": guide_info["regions"],
                    "stats": guide_info["stats"],
                })
            else:
                # guided run but this timestep is above the ALM start threshold
                alm_frames.append(None)
            v3_frames.append({
                "t": int(step["t"]),
                "selectedIdx": int(step["selected_idx"][0]),
                "pi": self._rounded(step["pi"][0]),
                "progress": self._rounded(step["progress"][0]),
                "center": self._rounded(center),
                "shape4": self._rounded(shape4),
                "regions": [],
            })
        # close the replay with the final x0 frame (same convention as V1/V2)
        p_history.append(p_history[-1]); e6_history.append(e6_history[-1])
        x0p_history.append(x0p_history[-1]); x0e6_history.append(x0e6_history[-1])
        alm_frames.append(None)
        v3_frames.append(dict(v3_frames[-1]))

        selected = int(result["selected_idx"][0])
        topology = (cands.coords[selected] if cands.num_valid
                    else np.zeros((self.cand_cfg.candidate_points, 2)))
        metrics = self._ellipse_metrics(occupancy, trace)
        metrics["trajCollision"] = bool(self._traj_collision(
            occupancy, np.asarray(p_history[-2])))
        metrics["regionCount"] = int(region_count)
        metrics["progressMonotonic"] = all(
            bool(np.all(np.diff(f["progress"]) >= -1e-6)) for f in v3_frames
            if f and f.get("progress"))
        selections = [int(f["selectedIdx"]) for f in v3_frames]
        metrics["stepJitter"] = int(sum(1 for a, b in zip(selections[:-1],
                                                          selections[1:]) if a != b))
        metrics["selectionChanges"] = int(len(set(selections)) > 1)

        return {
            "sampleKey": sample_key, "modelId": model_id, "seed": seed,
            "cacheHit": False, "condition": condition.tolist(), "obstacles": [],
            "stateLabels": ["t=%d" % int(step["t"]) for step in trace] + ["x0"],
            "schedule": {
                "sqrtAlphaBar": np.round(
                    self.schedule.sqrt_alphas_cumprod.cpu().numpy(), 8).tolist(),
                "sqrtOneMinusAlphaBar": np.round(
                    self.schedule.sqrt_one_minus_alphas_cumprod.cpu().numpy(),
                    8).tolist(),
            },
            "PHistory": p_history, "E6History": e6_history,
            "X0PHistory": x0p_history, "X0E6History": x0e6_history,
            # ALM overlay channel: convex regions + the pre-correction x0, or the
            # coarse backbone prediction when ALM guidance is disabled.
            "alm": {"enabled": bool(verify_regions),
                    "startT": int(self.alm_cfg.get("start_t", 7))
                    if verify_regions else 0,
                    "frames": alm_frames},
            "v2": {
                "engine": "v3-skeleton-dynamic",
                "epoch": epoch,
                "commitT": -1,
                "selection": "argmax",
                "numCandidates": int(cands.num_slots),
                "numValid": int(cands.num_valid),
                "rawK": int(cands.raw_k),
                "selectedIdx": selected,
                "pi": self._rounded(result["topology_pi"][0]),
                "topologyPath": self._rounded(topology),
                "candidatePaths": [self._rounded(cands.coords[m])
                                   for m in range(cands.num_slots)],
                "candidateMask": [bool(v) for v in cands.mask.tolist()],
                "candidateLengths": self._rounded(cands.lengths),
                "geometryPoints": int(packed["geometry_points"]),
                "nodePath": [int(v) for v in cands.node_paths[selected]]
                if cands.node_paths else [],
                "frames": v3_frames,
                "metrics": metrics,
                "skeleton": {"nodes": len(graph.nodes),
                             "branches": len(graph.branches)},
            },
        }


def _selftest(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    engine = V3Engine(device)
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
                      "numValid": v2["numValid"], "selectedIdx": v2["selectedIdx"],
                      "pi": v2["pi"], "metrics": v2["metrics"],
                      "skeleton": v2["skeleton"],
                      "steps": len(v2["frames"])}, indent=2))
    shapes = {k: np.asarray(payload[k]).shape for k in
              ("PHistory", "E6History", "X0PHistory", "X0E6History")}
    print("payload shapes:", shapes)
    print("E6 last[0]:", payload["E6History"][-1][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample", default="large-855")
    ap.add_argument("--model", default="v3_best")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--device", default=None)
    _selftest(ap.parse_args())

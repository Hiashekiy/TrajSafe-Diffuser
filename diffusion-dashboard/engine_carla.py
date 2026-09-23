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
  * the reverse loop is the REAL sampler state machine
    (WARMUP -> TRY_ACTIVATE -> GUIDED), so the dashboard shows exactly what
    inference does:

        x0_raw_history   the network's clean prediction BEFORE the ALM
        x0_history       the ALM SAFE clean prediction that DDIM consumed
        corridor         the frozen 128-region corridor + its bridge regions
        alm.frames       per-step violation before/after, curve correction, dual
        final_validation dense (512-point) collision / constraint / membership
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
from src.utils.checkpoint import load_model
from src.diffusion.sampler import ablation_configs, sample
from src.diffusion.schedule import NoiseSchedule
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import (CandidateConfig, generate_candidates,
                                         normalized_dtw)
from src.models.trajsafe import TrajSafePlanner

# ---------------------------------------------------------------- models
# The dashboard serves SEVERAL runs at once.  Every entry carries its OWN config
# because the historical-feedback arms have a different architecture
# (``model.feedback.enabled``): A/B2 were trained with feedback, B1 without, and
# the old 160k8 baseline on the k=8 cache.  The DISPLAYED data (occupancy / GT /
# candidates) comes from ``DASH_CONFIG`` (default: the 160k8p cache the campaign
# models were trained on), so switching the model does not switch the map.
#
#   A_oneshot    一步到位：从零直接开 feedback 两步 rollout（58 ep, best_task ep50）
#   B2_feedback  先基座再 feedback 微调（100 + 37 ep, best_task ep36）
#   B1_base      只训基座、feedback 关闭（100 ep, best_task ep24）——对照臂
#   REF_160k8    旧 160k8 (k=8) 的 best_task（跨缓存参考）
MODELS = {
    "A_oneshot": {
        "config": "configs/config_160k8p.yaml",
        "ckpt_dir": "outputs/campaign_a_oneshot/ckpt",
        "label": "A · 一步到位 feedback（best_task ep50）",
    },
    "B2_feedback": {
        "config": "configs/config_160k8p.yaml",
        "ckpt_dir": "outputs/campaign_b2_feedback/ckpt",
        "label": "B2 · 基座 + feedback 微调（best_task ep36）",
    },
    "B1_base": {
        "config": "configs/config_160k8p_s1.yaml",
        "ckpt_dir": "outputs/campaign_b1_base/ckpt",
        "label": "B1 · 基座（无 feedback，best_task ep24）",
    },
    "REF_160k8": {
        "config": "configs/config_160k8.yaml",
        "ckpt_dir": "outputs/bspline_carla_160k8/ckpt",
        "label": "REF · 旧 160k8 best_task（跨缓存）",
    },
    # Arm A's recipe retrained on the PRE-EROSION cache (k=0, free area 0.185):
    # 65 epochs / 2.4 h, best_task at epoch 59.  Only the dataset differs from
    # ``A_oneshot``, so it is the single-variable control for "how much of the
    # safety came from the widened k=8 channels".  It needs its OWN config
    # because that YAML points at data/carla_processed_160 (the architecture is
    # identical to A's, so the checkpoint files are drop-in interchangeable).
    "RAW160_oneshot": {
        "config": "configs/config_160raw_oneshot.yaml",
        "ckpt_dir": "outputs/oneshot_raw160/ckpt",
        "label": "RAW160 · 腐蚀前数据集上重训的 OneShot（best_task ep59）",
    },
}
# ``best`` is deliberately NOT offered for the campaign arms: their val-total
# best (epoch 7-26) is an overfit-topology artifact, ``best_task`` is the model
# that is actually deployed.
CKPT_KINDS = ("best_task", "latest")
CHECKPOINTS = {
    "%s:%s" % (run, kind): os.path.join(ROOT, spec["ckpt_dir"],
                                        "%s.pt" % kind)
    for run, spec in MODELS.items() for kind in CKPT_KINDS
}
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "config_160k8p.yaml")

# ---------------------------------------------------------------- datasets
# The SAME 420 test samples exist on two processed caches, so the panel can show
# how the identical sample/model/seed behaves when the free space is tighter:
#
#  160k8p  obstacles eroded by k=8 cells -> every channel widened by 10 m
#          (the cache the campaign models were trained and evaluated on)
#  160k4p  obstacles eroded by k=4 cells -> +5 m, built TEST-ONLY
#
# They are NOT nested near the crop border: ``--border-mode protect`` restores
# the outer k cells from the source occupancy, so at k=8 the ring 0..7 keeps the
# original obstacles while at k=4 it does not (72 270 cells differ, all inside
# 4 <= d <= 7 of the edge).  Elsewhere (d >= 8) k8p is strictly more permissive.
# Measured on test: free area 0.3573 (k8p) vs 0.2792 (k4p), straight start->goal
# line collides in 220/420 (k8p) vs 234/420 (k4p) samples.
DATASETS = {
    "raw160": {
        "root": "data/carla_processed_160",
        "label": "raw160 · 未腐蚀（障碍不动）",
        "note": "自由面积 0.185 · RAW160_oneshot 的训练缓存",
    },
    "160k8p": {
        "root": "data/carla_processed_160k8p",
        "label": "160k8p · 宽走廊（障碍各让 5 m）",
        "note": "campaign 训练/评估所用缓存 · 自由面积 0.357",
    },
    "160k4p": {
        "root": "data/carla_processed_160k4p",
        "label": "160k4p · 紧走廊（障碍各让 2.5 m）",
        "note": "更严格的地图 · 自由面积 0.279 · 仅 test",
    },
}
DEFAULT_DATASET = "160k8p"
SPLIT_ORDER = ("train", "val", "test")


def dataset_key(dataset_id=None) -> str:
    """Validate a dataset id (``None`` -> the default cache)."""
    key = str(dataset_id or DEFAULT_DATASET).strip().lower()
    if key not in DATASETS:
        raise ValueError("未知数据集 %r（可选：%s）"
                         % (dataset_id, ", ".join(DATASETS)))
    return key


def dataset_root(dataset_id=None):
    """(id, absolute processed root) for a dataset id."""
    key = dataset_key(dataset_id)
    return key, os.path.abspath(os.path.join(ROOT, DATASETS[key]["root"]))


def dataset_splits(dataset_id=None):
    """(id, splits actually present on disk) - 160k4p ships test only, so the
    panel must not offer train/val for it."""
    key, root = dataset_root(dataset_id)
    return key, tuple(name for name in SPLIT_ORDER
                      if os.path.isdir(os.path.join(root, name)))


def dataset_catalog():
    """Everything the front-end needs to render the dataset selector."""
    out = []
    for key in DATASETS:
        _, root = dataset_root(key)
        _, splits = dataset_splits(key)
        out.append({"id": key, "label": DATASETS[key]["label"],
                    "note": DATASETS[key].get("note", ""),
                    "root": root, "splits": list(splits)})
    return out


def model_config(model_id, cache={}):
    """The YAML a given model id must be BUILT with (architecture mismatch)."""
    run = str(model_id).split(":", 1)[0]
    if run not in MODELS:
        raise ValueError("unknown checkpoint %r" % model_id)
    path = os.path.join(ROOT, MODELS[run]["config"])
    if path not in cache:
        cache[path] = load_config(path)
    return cache[path]


def model_label(model_id):
    run = str(model_id).split(":", 1)[0]
    spec = MODELS.get(run) or {}
    return "%s · %s" % (spec.get("label", run), model_id.split(":", 1)[-1])

# ALM stat keys that are forwarded to the dashboard (floats only).
ALM_STAT_KEYS = (
    "max_violation_before", "max_violation_after",
    "mean_positive_violation_before", "mean_positive_violation_after",
    "constraint_feasible_rate", "active_constraint_count",
    "mean_curve_correction_scene", "max_curve_correction_scene",
    "mean_curve_correction_m", "max_curve_correction_m",
    "lambda_mean", "lambda_max", "inner_steps_used",
)


class Engine:
    def __init__(self, device, processed_root=None, config_path=None,
                 dataset=None):
        self.device = device
        # ``DASH_CONFIG`` (or the default 160k8p config) decides the GEOMETRY
        # (knots / candidates / corridor knobs) shared by every cached dataset;
        # each model still builds itself from its own YAML.
        self.config_path = os.path.abspath(
            config_path or os.environ.get("DASH_CONFIG", DEFAULT_CONFIG))
        print("[engine] display config: %s" % self.config_path, flush=True)
        cfg = load_config(self.config_path)
        self.cfg = cfg
        # ``dataset`` selects the displayed cache per request; ``processed_root``
        # (or DASH_DATASET) only overrides the DEFAULT dataset's root, which is
        # what the single-dataset deployments used to pass.
        self.dataset_id = dataset_key(
            dataset or os.environ.get("DASH_DATASET") or DEFAULT_DATASET)
        if processed_root:
            self.processed_root = os.path.abspath(processed_root)
        else:
            _, self.processed_root = dataset_root(self.dataset_id)
        cfg_root = cfg["data"].get("processed_root")
        if cfg_root and os.path.abspath(cfg_root) != self.processed_root:
            # The dataset registry - not the YAML - decides which cache is
            # served, otherwise the selector would lie about what is displayed.
            print("[engine] note: config data.processed_root=%s is ignored; "
                  "dataset %s comes from the registry"
                  % (os.path.abspath(cfg_root), self.dataset_id), flush=True)
        print("[engine] default dataset: %s -> %s (%s)"
              % (self.dataset_id, self.processed_root,
                 ", ".join(dataset_splits(self.dataset_id)[1])), flush=True)
        self.geometry_points = int((cfg.get("topology") or {}).get(
            "candidate_geometry_points", 1280))
        self.cand_cfg = CandidateConfig.from_dict(cfg.get("topology") or {},
                                                  strict=False)
        self.skel_cfg = cfg.get("skeleton") or {}
        self.alm_cfg = cfg.get("alm") or {}
        self.corridor_cfg = cfg.get("corridor") or {}
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
    def dataset_dir(self, dataset=None):
        """(id, root) for one request; the default dataset honours the
        ``processed_root`` / ``DASH_CONFIG`` override from the constructor."""
        key = dataset_key(dataset)
        if key == self.dataset_id:
            return key, self.processed_root
        return dataset_root(key)

    def split_data(self, split, dataset=None):
        """Arrays of one (dataset, split), cached per pair.

        ``160k4p`` only ships ``test``; asking for train/val raises a readable
        error instead of an np.load traceback.
        """
        key, root = self.dataset_dir(dataset)
        cache_key = (key, str(split))
        if cache_key not in self._splits:
            d = os.path.join(root, str(split))
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    "数据集 %s 没有 %r 划分（%s）; 可用: %s"
                    % (key, str(split), d, ", ".join(dataset_splits(key)[1])))
            control_gt = os.path.join(d, "control_gt.npy")
            self._splits[cache_key] = {
                "dataset": key,
                "conditions": np.load(os.path.join(d, "conditions.npy")),
                "curve_gt": np.load(os.path.join(d, "curve_gt.npy")),
                "occupancy": np.load(os.path.join(d, "occupancy.npy"),
                                     mmap_mode="r"),
                "control_gt": (np.load(control_gt, mmap_mode="r")
                               if os.path.exists(control_gt) else None),
            }
        return self._splits[cache_key]

    def split_control_gt(self, split, index, dataset=None):
        """The dataset's 32-control GT polygon for one sample (may be absent)."""
        data = self.split_data(split, dataset)
        if data["control_gt"] is None:
            return None
        return np.asarray(data["control_gt"][int(index)], dtype=np.float32)

    def sample_arrays(self, split, index, dataset=None):
        data = self.split_data(split, dataset)
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
            # arch='auto': a checkpoint written before the control-space
            # refactor replays the legacy 128-curve-token chain unchanged.
            # The MODEL config (feedback architecture, C, knots) comes from the
            # run itself; the DISPLAYED data still comes from self.cfg.
            cfg = model_config(model_id)
            model, ckpt, _ = load_model(cfg, path, arch="auto",
                                        device=self.device)
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
    @staticmethod
    def _alm_stats_row(stats, b):
        if stats is None:
            return None
        return {k: float(stats[k][b]) for k in ALM_STAT_KEYS if k in stats}

    def _run_sampler(self, model, cond, occ, packed, seed, alm_cfg,
                     corridor_cfg, steps=None, times=None):
        """Run the production sampler (WARMUP / TRY_ACTIVATE / GUIDED).

        ``steps`` = number of reverse forwards (uniform sub-sampling of the 16
        training timesteps, ``None`` = all of them); ``times`` = an explicit
        non-uniform schedule (see :func:`src.diffusion.sampler.sample`).
        """
        return sample(
            model, self.schedule, cond, occ, packed["xy"], packed["mask"],
            packed["geometry"], packed["geometry_lengths"],
            device=self.device, steps=steps, times=times, seed=seed,
            return_trace=True, alm_config=alm_cfg, corridor_config=corridor_cfg)

    def _steps_from_result(self, out):
        """Map the sampler trace onto the dashboard's per-frame records."""
        steps = []
        for tr in out["trace"]:
            steps.append({
                "t": int(tr["t"]), "s": int(tr["s"]),
                "state": tr["p"][0],               # decoded noisy state P_t
                "x0": tr["p_safe"][0],             # ALM safe x0 (DDIM input)
                "x0_raw": tr["p_raw"][0],          # network x0 before the ALM
                "coarse": tr["coarse"][0],
                "control": tr["q0_safe"][0],
                "control_raw": tr["q0_raw"][0],
                "selected_idx": int(tr["selected_idx"][0]),
                "pi": tr["pi"][0],
                "guided": bool(tr["guided"][0]),
                "alm_active": bool(tr["alm_active"]),
                "alm_stats": tr["alm_stats"],
                "ellipse": {
                    "center": tr["ellipse_center"],
                    "shape4": tr["ellipse_shape4"],
                    "a": tr["ellipse_a"], "b": tr["ellipse_b"],
                    "theta": tr["ellipse_theta"],
                    "progress": tr["progress"],
                },
            })
        last = steps[-1]
        # close the replay with the terminal x0 frame
        steps.append({**last, "t": -1, "state": last["x0"]})
        return steps

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
    def generate(self, sample_key, split, index, occupancy, condition, seed,
                 model_id="best_task", verify_regions=False, alm_enabled=True,
                 ablation=None, steps=None, times=None, dataset=None):
        model, epoch = self.get_model(model_id)
        dkey = dataset_key(dataset)
        condition = np.asarray(condition, dtype=np.float32).reshape(2, 2)
        occupancy = np.asarray(occupancy, dtype=np.float32)
        graph = self.get_graph(
            "%s:%s:%d:%s" % (dkey, split, int(index),
                             self.occupancy_key(occupancy)),
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

        alm_cfg, corridor_cfg = dict(self.alm_cfg), dict(self.corridor_cfg)
        if steps is not None:
            # Warm-up is counted in EXECUTED forwards, not in timesteps: with a
            # short schedule the configured 3 would swallow the guided phase
            # (4 forwards = at most 1 warm-up + 3 guided), so clamp it to keep at
            # least 2 guided forwards.  Measured on the full test split, 4 steps
            # with warm-up 1/2/3 is indistinguishable from the 16-step baseline.
            base = int(alm_cfg.get("warmup_reverse_steps", 3))
            alm_cfg["warmup_reverse_steps"] = max(1, min(base, int(steps) - 2))
        if ablation:
            alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg,
                                                     ablation)
        elif not alm_enabled:
            alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg, "A")

        with torch.no_grad():
            out = self._run_sampler(model, cond, occ, packed, seed, alm_cfg,
                                    corridor_cfg, steps=steps, times=times)
        steps = self._steps_from_result(out)

        state_history, x0_history, x0_raw_history = [], [], []
        control_history, control_raw_history = [], []
        ellipse_history, alm_frames, topology_frames = [], [], []
        for k, st in enumerate(steps):
            ell = st["ellipse"]
            center = ell["center"][0].detach().cpu().numpy()
            shape4 = ell["shape4"][0].detach().cpu().numpy()
            state_history.append(self._rounded(st["state"]))
            x0_history.append(self._rounded(st["x0"]))
            x0_raw_history.append(self._rounded(st["x0_raw"]))
            control_history.append(self._rounded(st["control"]))
            control_raw_history.append(self._rounded(st["control_raw"]))
            ellipse_history.append({"center": self._rounded(center),
                                    "shape4": self._rounded(shape4)})
            alm_frames.append({
                "t": int(st["t"]),
                "raw_p": self._rounded(st["x0_raw"]),
                "safe_p": self._rounded(st["x0"]),
                "guided": bool(st["guided"]),
                "alm_active": bool(st["alm_active"]),
                "enforced": [],
                "regions": [],
                "stats": self._alm_stats_row(st["alm_stats"], 0) or {},
            })
            topology_frames.append({
                "t": int(st["t"]),
                "selected_idx": int(st["selected_idx"]),
                "pi": self._rounded(st["pi"]),
                "progress": self._rounded(
                    model.fixed_progress[None].expand(
                        1, int(model.fixed_progress.numel()))[0]),
                "center": self._rounded(center),
                "shape4": self._rounded(shape4),
                "regions": [],
            })
        labels = ["t=%d" % int(st["t"]) for st in steps[:-1]] + ["x0"]

        selected = int(steps[-1]["selected_idx"])
        final_curve = np.asarray(x0_history[-1], dtype=np.float64)
        raw_curve = np.asarray(x0_raw_history[-1], dtype=np.float64)
        metrics = self._ellipse_metrics(occupancy, steps)
        metrics["traj_collision"] = bool(
            not self._free_mask(occupancy, final_curve).all())
        metrics["raw_traj_collision"] = bool(
            not self._free_mask(occupancy, raw_curve).all())
        metrics["region_count"] = int(
            (out["corridors"][0] or {}).get("num_cells", 0))
        metrics["progress_monotonic"] = True
        metrics["alm_status"] = out["alm_status"][0]
        metrics["activation_step"] = int(out["activation_step"][0])
        metrics["frozen_topology_idx"] = int(out["frozen_topology_idx"][0])
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

        activation = dict(out["activation_info"])
        activation["step"] = int(out["activation_step"][0])
        activation["frozen_topology_idx"] = int(out["frozen_topology_idx"][0])
        activation["status"] = out["alm_status"][0]
        activation["guided"] = bool(out["guided"][0])

        return {
            "sample_key": sample_key, "model_id": model_id, "seed": int(seed),
            "cache_hit": False, "condition": condition.tolist(),
            "dataset": dkey,
            "dataset_label": DATASETS[dkey]["label"],
            "dataset_root": self.dataset_dir(dkey)[1],
            "state_labels": labels,
            # reverse forwards actually executed (t >= 0 frames; the trailing
            # ``-1`` frame is the terminal x0), plus the schedule they sample
            "steps": int(sum(1 for st in steps if int(st["t"]) >= 0)),
            "times": [int(st["t"]) for st in steps],
            "schedule": {
                "sqrt_alpha_bar": np.round(
                    self.schedule.sqrt_alphas_cumprod.cpu().numpy(), 8).tolist(),
                "sqrt_one_minus_alpha_bar": np.round(
                    self.schedule.sqrt_one_minus_alphas_cumprod.cpu().numpy(),
                    8).tolist(),
            },
            "state_history": state_history,
            # the ALM SAFE clean prediction: this is what DDIM consumed
            "x0_history": x0_history,
            # the network's raw clean prediction before the ALM correction
            "x0_raw_history": x0_raw_history,
            "control_history": control_history,
            "control_raw_history": control_raw_history,
            "ellipse_history": ellipse_history,
            "alm": {
                "enabled": bool(out["alm_settings"]["enabled"]),
                "mode": out["alm_settings"]["mode"],
                "start_t": 0,
                "warmup_reverse_steps":
                    out["alm_settings"]["warmup_reverse_steps"],
                "activation_step": int(out["activation_step"][0]),
                "frozen_topology_idx": int(out["frozen_topology_idx"][0]),
                "status": out["alm_status"][0],
                "rho": out["alm_settings"]["rho"],
                "constraint_tol": out["alm_settings"]["constraint_tol"],
                "max_curve_step_scene":
                    out["alm_settings"]["max_curve_step_scene"],
                "frames": alm_frames,
            },
            "corridor": out["corridors"][0],
            "activation": activation,
            "pack_summary": out["pack_summary"],
            "final_validation": out["final_validation"][0],
            "progress_alignment": out["progress_alignment"][0],
            "topology": {
                "epoch": epoch,
                "selection": "frozen" if bool(out["guided"][0]) else "argmax",
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

"""Local GPU inference and cache service for the diffusion dashboard."""
from __future__ import annotations

import json
import hashlib
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch


SITE_ROOT = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SITE_ROOT, ".."))
sys.path.insert(0, ROOT)

from src.diffusion.schedule import NoiseSchedule
from src.diffusion.alm_guidance import alm_correct
from src.geometry.convex_corridor import EllipseRegionBuilder
from src.geometry.ellipse_center_repair import EllipseCenterRepair
from src.models.joint import JointPlanner
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


CHECKPOINTS = {
    "epoch100": "outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced/epoch_100.pt",
    "continue100": "outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced_continue100/best.pt",
    "continue200": "outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced_continue200/best.pt",
}
MAZES = ("umaze", "medium", "large")
CACHE_DIR = os.path.join(SITE_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

cfg = load_config(os.path.join(ROOT, "configs", "config_v1_continue.yaml"))
alm_cfg = load_config(os.path.join(ROOT, "configs", "config_v1_alm.yaml"))["alm"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = os.path.join(ROOT, "data", "processed_scene_v1", "test")
conditions = np.load(os.path.join(test_dir, "conditions.npy"))
maze_ids = np.load(os.path.join(test_dir, "maze_id.npy"))
maps = {name: np.load(os.path.join(ROOT, "data", "processed_scene_v1", "maps", f"{name}.npy")) for name in MAZES}
schedule = NoiseSchedule(cfg["diffusion"]["timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"]).to(device)
models: dict[str, JointPlanner] = {}
inference_lock = threading.Lock()

with open(os.path.join(SITE_ROOT, "lib", "dashboard-catalog.json"), "r", encoding="utf-8") as handle:
    catalog = json.load(handle)
sample_lookup = {sample["key"]: sample for sample in catalog["samples"]}


def get_model(model_id: str):
    if model_id not in models:
        model = JointPlanner(cfg["model"]).to(device)
        load_checkpoint(os.path.join(ROOT, CHECKPOINTS[model_id]), model, map_location=device)
        model.eval()
        models[model_id] = model
    return models[model_id]


def rounded(tensor: torch.Tensor):
    return np.round(tensor.detach().cpu().numpy().astype(np.float64), 5).tolist()


def polygon_vertices(A, b, mask, tol=1e-5):
    """Return ordered vertices of a bounded 2-D halfspace intersection."""
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


@torch.no_grad()
def generate_alm_trace(request):
    sample_key = request.get("sampleKey")
    if sample_key not in sample_lookup:
        raise ValueError("未知样本")
    raw_p = np.asarray(request.get("rawP"), dtype=np.float32)
    raw_e = np.asarray(request.get("rawE6"), dtype=np.float32)
    if raw_p.shape != (128, 2) or raw_e.shape != (128, 6):
        raise ValueError("ALM trace 需要 [128,2] 的 P 和 [128,6] 的 E6")
    if not np.isfinite(raw_p).all() or not np.isfinite(raw_e).all():
        raise ValueError("ALM trace 输入包含非有限值")

    sample = sample_lookup[sample_key]
    dataset_id = int(sample["datasetId"])
    condition = np.asarray(request.get("condition") or conditions[dataset_id], dtype=np.float32)
    maze = MAZES[int(maze_ids[dataset_id])]
    occupancy = maps[maze].copy()
    for obstacle in request.get("obstacles") or []:
        ox, oy, radius = map(float, obstacle)
        cx, cy, pr = (ox + 1) * 127.5, (oy + 1) * 127.5, radius * 127.5
        yy, xx = np.ogrid[:256, :256]
        occupancy[(xx - cx) ** 2 + (yy - cy) ** 2 <= pr ** 2] = 1.0

    p = torch.as_tensor(raw_p[None], dtype=torch.float32, device=device)
    e = torch.as_tensor(raw_e[None], dtype=torch.float32, device=device)
    cond = torch.as_tensor(condition[None], dtype=torch.float32, device=device)
    p[:, 0], p[:, -1] = cond[:, 0], cond[:, 1]
    map_tensor = torch.as_tensor(
        occupancy, dtype=torch.float32, device=device)[None, None]
    builder = EllipseRegionBuilder(map_tensor, alm_cfg)
    repair = EllipseCenterRepair(builder)(p, e, cond[:, 0])

    trace = []
    A, b = repair.A[:, 1:], repair.b[:, 1:]
    face_mask, valid = repair.face_mask[:, 1:], repair.valid[:, 1:]
    alm_p, _, stats = alm_correct(
        p, A, b, face_mask, valid,
        torch.zeros(1, p.shape[1] - 1, dtype=p.dtype, device=device),
        float(alm_cfg.get("rho", 5.0)),
        step_size=float(alm_cfg.get("step_size", 0.03)),
        inner_steps=int(alm_cfg.get("inner_steps", 4)),
        max_grad_norm=float(alm_cfg.get("max_grad_norm", 1.0)),
        max_correction_per_step=float(alm_cfg.get("max_correction_per_step", 0.10)),
        collision_fn=builder.segment_needs_guidance,
        collect_stats=True,
        trace_callback=lambda inner, value: trace.append({
            "inner": inner, "P": rounded(value[0])}),
    )

    cpu_A = repair.A[0].cpu().numpy()
    cpu_b = repair.b[0].cpu().numpy()
    cpu_mask = repair.face_mask[0].cpu().numpy()
    cpu_valid = repair.valid[0].cpu().numpy()
    regions = [
        polygon_vertices(cpu_A[k], cpu_b[k], cpu_mask[k]) if cpu_valid[k] else []
        for k in range(len(cpu_valid))
    ]
    scalar_stats = {key: float(value.cpu()) for key, value in {
        **repair.stats, **stats}.items()}
    return {
        "sampleKey": sample_key,
        "rawTrajectory": rounded(p[0]),
        "finalTrajectory": rounded(alm_p[0]),
        "rawCenters": rounded(p[0] + e[0, :, :2]),
        "repairedCenters": rounded(repair.centers[0]),
        "ellipseShape": rounded(e[0, :, 2:]),
        "regionValid": cpu_valid.tolist(),
        "regions": regions,
        "trajectoryFrames": trace,
        "stats": scalar_stats,
    }


@torch.no_grad()
def generate(sample_key: str, model_id: str, seed: int, custom_condition=None, obstacles=None):
    sample = sample_lookup[sample_key]
    dataset_id = int(sample["datasetId"])
    condition = np.asarray(custom_condition if custom_condition is not None else conditions[dataset_id], dtype=np.float32)
    if condition.shape != (2, 2) or not np.isfinite(condition).all() or np.abs(condition).max() > 1:
        raise ValueError("起终点必须是 [-1,1]² 内的两个坐标")
    cond = torch.as_tensor(condition[None], dtype=torch.float32, device=device)
    maze = MAZES[int(maze_ids[dataset_id])]
    occupancy = maps[maze].copy()
    for obstacle in obstacles or []:
        if len(obstacle) != 3: raise ValueError("障碍点格式错误")
        ox, oy, radius = map(float, obstacle)
        if not (-1 <= ox <= 1 and -1 <= oy <= 1 and 0.01 <= radius <= 0.25): raise ValueError("障碍点超出地图或半径无效")
        cx, cy, pr = (ox + 1) * 127.5, (oy + 1) * 127.5, radius * 127.5
        yy, xx = np.ogrid[:256, :256]
        occupancy[(xx - cx) ** 2 + (yy - cy) ** 2 <= pr ** 2] = 1.0
    map_tensor = torch.as_tensor(occupancy, dtype=torch.float32, device=device)[None, None]
    model = get_model(model_id)
    torch.manual_seed(seed)
    start, goal = cond[:, 0], cond[:, 1]
    p = torch.randn(1, model.horizon, 2, device=device)
    e = torch.randn(1, model.horizon, 6, device=device)
    p[:, 0], p[:, -1] = start, goal
    sqrt_ab = schedule.sqrt_alphas_cumprod.detach().cpu().tolist()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod.detach().cpu().tolist()
    p_history, e_history, x0_p_history, x0_e_history, labels = [], [], [], [], []
    for t in reversed(range(schedule.num_timesteps)):
        p_history.append(rounded(p[0])); e_history.append(rounded(e[0])); labels.append(f"t={t}")
        tb = torch.full((1,), t, device=device, dtype=torch.long)
        ab = torch.full((1,), float(sqrt_ab[t]), device=device)
        out = model(p, e, map_tensor, cond, tb, ab)
        x0_p, x0_e = out["x0_p"], out["x0_e"]
        x0_p[:, 0], x0_p[:, -1] = start, goal
        x0_p_history.append(rounded(x0_p[0]))
        x0_e_history.append(rounded(x0_e[0]))
        if t == 0:
            p, e = x0_p, x0_e
        else:
            eps_p = (p - float(sqrt_ab[t]) * x0_p) / float(sqrt_1ma[t])
            eps_e = (e - float(sqrt_ab[t]) * x0_e) / float(sqrt_1ma[t])
            p = float(sqrt_ab[t - 1]) * x0_p + float(sqrt_1ma[t - 1]) * eps_p
            e = float(sqrt_ab[t - 1]) * x0_e + float(sqrt_1ma[t - 1]) * eps_e
        p[:, 0], p[:, -1] = start, goal
    p_history.append(rounded(p[0])); e_history.append(rounded(e[0])); labels.append("x0")
    x0_p_history.append(rounded(p[0])); x0_e_history.append(rounded(e[0]))
    return {
        "sampleKey": sample_key, "modelId": model_id, "seed": seed, "cacheHit": False,
        "condition": condition.tolist(), "obstacles": obstacles or [],
        "stateLabels": labels,
        "schedule": {
            "sqrtAlphaBar": np.round(schedule.sqrt_alphas_cumprod.cpu().numpy(), 8).tolist(),
            "sqrtOneMinusAlphaBar": np.round(schedule.sqrt_one_minus_alphas_cumprod.cpu().numpy(), 8).tolist(),
        },
        "PHistory": p_history, "E6History": e_history,
        "X0PHistory": x0_p_history, "X0E6History": x0_e_history,
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers(); self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ready", "device": str(device), "cached": len(os.listdir(CACHE_DIR))})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route not in ("/generate", "/alm-trace"):
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if route == "/alm-trace":
                with inference_lock:
                    result = generate_alm_trace(request)
                return self.send_json(200, result)
            sample_key, model_id, seed = request.get("sampleKey"), request.get("modelId"), int(request.get("seed", 42))
            custom_condition = request.get("condition")
            obstacles = request.get("obstacles") or []
            if sample_key not in sample_lookup: raise ValueError("未知样本")
            if model_id not in CHECKPOINTS: raise ValueError("未知模型")
            if seed < 0 or seed > 2_147_483_647: raise ValueError("seed 超出范围")
            cache_payload = json.dumps({"format": 2, "sample": sample_key, "model": model_id, "seed": seed, "condition": custom_condition, "obstacles": obstacles}, sort_keys=True, separators=(",", ":"))
            cache_key = hashlib.sha1(cache_payload.encode()).hexdigest()[:12]
            cache_path = os.path.join(CACHE_DIR, f"{model_id}__{sample_key}__seed{seed}__{cache_key}.json")
            started = time.perf_counter()
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as handle: result = json.load(handle)
                result["cacheHit"] = True
            else:
                with inference_lock:
                    if os.path.exists(cache_path):
                        with open(cache_path, "r", encoding="utf-8") as handle: result = json.load(handle)
                        result["cacheHit"] = True
                    else:
                        result = generate(sample_key, model_id, seed, custom_condition, obstacles)
                        with open(cache_path, "w", encoding="utf-8") as handle: json.dump(result, handle, separators=(",", ":"))
            result["elapsedMs"] = round((time.perf_counter() - started) * 1000, 1)
            self.send_json(200, result)
        except Exception as error:
            self.send_json(400, {"error": str(error)})

    def log_message(self, fmt, *args):
        print(f"[dashboard-api] {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"Diffusion dashboard API on http://localhost:8765 ({device})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()

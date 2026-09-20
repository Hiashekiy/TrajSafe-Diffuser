"""Local GPU inference and cache service for the diffusion dashboard.

This server exposes the report-faithful **TrajSafe-Diffuser**:

    P_t -> H_traj -> {R_m} -> m = argmax(pi) -> H_prog -> s
        -> c = Gamma_m(s) -> H_ell -> H_clean -> P0_hat -> DDIM

The model class, the online skeleton/candidate search, the sampler and the
optional inference-time ALM guidance all live in ``engine.Engine``.  This
module only owns the HTTP layer, the per-request obstacle overlay and the JSON
cache.

    python backend.py     # http://localhost:8765
"""
from __future__ import annotations

import hashlib
import json
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

from engine import CHECKPOINTS, Engine


MAZES = ("umaze", "medium", "large")
CACHE_DIR = os.path.join(SITE_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FORMAT = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maps = {name: np.load(os.path.join(ROOT, "data", "scenes",
                                   "maps", f"{name}.npy")) for name in MAZES}
test_dir = os.path.join(ROOT, "data", "scenes", "test")
conditions = np.load(os.path.join(test_dir, "conditions.npy"))

with open(os.path.join(SITE_ROOT, "lib", "dashboard-catalog.json"), "r",
          encoding="utf-8") as handle:
    catalog = json.load(handle)
sample_lookup = {sample["key"]: sample for sample in catalog["samples"]}

engine = None
inference_lock = threading.Lock()


def get_engine() -> Engine:
    """Lazily build the inference engine (own config, model class and sampler)."""
    global engine
    if engine is None:
        engine = Engine(device)
    return engine


def clear_generation_cache():
    """Remove prior generated JSON files before every diffusion run.

    The cache directory is fixed relative to this backend. Only regular JSON
    files directly inside that directory are removed; subdirectories and other
    files are never touched.
    """
    removed = 0
    for entry in os.scandir(CACHE_DIR):
        if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
            os.remove(entry.path)
            removed += 1
    return removed


def apply_obstacles(maze: str, obstacles):
    """Base occupancy of a maze plus the user drawn circular obstacles."""
    occupancy = maps[maze].copy()
    for obstacle in obstacles or []:
        if len(obstacle) != 3:
            raise ValueError("障碍点格式错误")
        ox, oy, radius = map(float, obstacle)
        if not (-1 <= ox <= 1 and -1 <= oy <= 1 and 0.01 <= radius <= 0.25):
            raise ValueError("障碍点超出地图或半径无效")
        cx, cy, pr = (ox + 1) * 127.5, (oy + 1) * 127.5, radius * 127.5
        yy, xx = np.ogrid[:256, :256]
        occupancy[(xx - cx) ** 2 + (yy - cy) ** 2 <= pr ** 2] = 1.0
    return occupancy


@torch.no_grad()
def generate(sample_key: str, model_id: str, seed: int, custom_condition=None,
             obstacles=None, alm_enabled: bool | None = None):
    """Run one reverse diffusion and return the dashboard payload."""
    sample = sample_lookup[sample_key]
    dataset_id = int(sample["datasetId"])
    condition = np.asarray(
        custom_condition if custom_condition is not None else conditions[dataset_id],
        dtype=np.float32)
    if condition.shape != (2, 2) or not np.isfinite(condition).all() or np.abs(condition).max() > 1:
        raise ValueError("起终点必须是 [-1,1]² 内的两个坐标")
    engine = get_engine()
    if alm_enabled is None:
        alm_enabled = bool(engine.alm_cfg.get("enabled", False))
    # Candidates are generated ONLINE on the current occupancy (including user
    # drawn obstacles) with the same generator as the offline preprocessing, so
    # every candidate search path is returned for the UI to draw.
    payload = engine.generate(
        sample_key, dataset_id, sample["maze"],
        apply_obstacles(sample["maze"], obstacles), condition, seed,
        model_id=model_id, verify_regions=bool(alm_enabled))
    payload["obstacles"] = obstacles or []
    return payload


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ready", "device": str(device),
                                 "cached": len(os.listdir(CACHE_DIR))})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            sample_key = request.get("sample_key")
            model_id = request.get("model_id")
            seed = int(request.get("seed", 42))
            custom_condition = request.get("condition")
            obstacles = request.get("obstacles") or []
            alm_enabled = bool(request.get("alm_enabled", True))
            if sample_key not in sample_lookup:
                raise ValueError("未知样本")
            if model_id not in CHECKPOINTS:
                raise ValueError("未知模型")
            if seed < 0 or seed > 2_147_483_647:
                raise ValueError("seed 超出范围")
            cache_payload = json.dumps(
                {"format": CACHE_FORMAT, "alm": alm_enabled, "sample": sample_key,
                 "model": model_id, "seed": seed, "condition": custom_condition,
                 "obstacles": obstacles},
                sort_keys=True, separators=(",", ":"))
            cache_key = hashlib.sha1(cache_payload.encode()).hexdigest()[:12]
            cache_path = os.path.join(
                CACHE_DIR, f"{model_id}__{sample_key}__seed{seed}__{cache_key}.json")
            started = time.perf_counter()
            with inference_lock:
                clear_generation_cache()
                result = generate(sample_key, model_id, seed, custom_condition,
                                  obstacles, alm_enabled)
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, separators=(",", ":"))
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.send_json(200, result)
        except Exception as error:
            self.send_json(400, {"error": str(error)})

    def log_message(self, fmt, *args):
        print(f"[dashboard-api] {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"TrajSafe dashboard API on http://localhost:8765 ({device})",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()

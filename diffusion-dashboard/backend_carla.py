"""Local GPU inference service for the CARLA Diffusion Lens dashboard.

Same HTTP contract as the legacy ``backend.py`` (GET /health, POST /generate),
but the engine is :mod:`engine_carla`:

    Q_t [1,32,2] --(fixed B-spline decode)--> P_t [1,128,2] --> network
    -> argmax(pi) candidate -> fixed Skeleton centres c_i = Gamma_m(i/127)
    -> 128 ellipses -> x0 controls -> decode -> curve

ALM guidance is NOT migrated to control space, so ``alm_enabled`` is accepted
for compatibility but the response always reports ``alm.enabled = false`` and
carries no convex regions.

    python backend_carla.py       # http://localhost:8765
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
sys.path.insert(0, SITE_ROOT)

from engine_carla import CHECKPOINTS, Engine

CACHE_DIR = os.path.join(SITE_ROOT, "cache-carla")
os.makedirs(CACHE_DIR, exist_ok=True)
# Bump whenever the payload schema changes: it is part of the cache key, so a
# restarted backend can never replay a payload produced by older code.
PAYLOAD_FORMAT = 3
CACHE_FORMAT = PAYLOAD_FORMAT
CATALOG_PATH = os.path.join(SITE_ROOT, "lib", "dashboard-catalog-carla.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
    catalog = json.load(handle)
sample_lookup = {sample["key"]: sample for sample in catalog["samples"]}
occupancy_cache = {}


def resolve_sample(sample_key: str) -> int:
    """sample_key -> dataset index.

    Accepts the catalog keys ("test_0018") and, so that a stale front-end bundle
    or a hand-typed key can never break the panel, any "<name>_<digits>" key
    whose digits are a valid index of the processed split.
    """
    if sample_key in sample_lookup:
        return int(sample_lookup[sample_key]["datasetId"])
    digits = "".join(ch for ch in str(sample_key) if ch.isdigit())
    if digits:
        index = int(digits)
        n = len(get_engine().split_data("test")["conditions"])
        if 0 <= index < n:
            return index
    raise ValueError("未知样本 %r（可选：%s …）"
                     % (sample_key, ", ".join(sorted(sample_lookup)[:4])))


def sample_occupancy(index: int) -> np.ndarray:
    if index not in occupancy_cache:
        engine = get_engine()
        occupancy_cache[index] = engine.sample_arrays("test", index)[0]
        if len(occupancy_cache) > 64:
            occupancy_cache.pop(next(iter(occupancy_cache)))
    return occupancy_cache[index].copy()


def apply_obstacles(occupancy: np.ndarray, obstacles):
    """Base occupancy plus the user drawn circular obstacles (scene coords)."""
    for obstacle in obstacles or []:
        if len(obstacle) != 3:
            raise ValueError("障碍点格式错误")
        ox, oy, radius = map(float, obstacle)
        if not (-1 <= ox <= 1 and -1 <= oy <= 1 and 0.01 <= radius <= 0.25):
            raise ValueError("障碍点超出地图或半径无效")
        cx, cy, pr = (ox + 1) * 127.5, (oy + 1) * 127.5, radius * 127.5
        yy, xx = np.ogrid[:occupancy.shape[0], :occupancy.shape[1]]
        occupancy[(xx - cx) ** 2 + (yy - cy) ** 2 <= pr ** 2] = 1.0
    return occupancy


engine = None
inference_lock = threading.Lock()


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = Engine(device)
    return engine


def clear_generation_cache():
    removed = 0
    for entry in os.scandir(CACHE_DIR):
        if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
            os.remove(entry.path)
            removed += 1
    return removed


@torch.no_grad()
def generate(sample_key: str, model_id: str, seed: int, custom_condition=None,
             obstacles=None, alm_enabled: bool | None = None):
    index = resolve_sample(sample_key)
    eng = get_engine()
    base = sample_occupancy(index)
    default_condition = eng.split_data("test")["conditions"][index]
    condition = np.asarray(
        custom_condition if custom_condition is not None else default_condition,
        dtype=np.float32)
    if condition.shape != (2, 2) or not np.isfinite(condition).all() \
            or np.abs(condition).max() > 1:
        raise ValueError("起终点必须是 [-1,1]² 内的两个坐标")
    occupancy = apply_obstacles(base, obstacles)
    payload = eng.generate(sample_key, "test", index, occupancy, condition,
                           seed, model_id=model_id,
                           verify_regions=False)
    payload["obstacles"] = obstacles or []
    payload["format"] = PAYLOAD_FORMAT
    payload["alm_note"] = ("B-spline control-space ALM 尚未迁移，本面板只跑"
                           "报告一致的 DDIM（无凸区域修正）")
    return payload


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {
                "status": "ready", "device": str(device),
                "engine": "carla-controlspace-32",
                "format": PAYLOAD_FORMAT,
                "features": ["control_history", "topology", "ellipse_history"],
                "checkpoints": sorted(CHECKPOINTS.keys()),
                "samples": len(sample_lookup),
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
            resolve_sample(sample_key)
            if model_id not in CHECKPOINTS:
                raise ValueError("未知模型")
            if seed < 0 or seed > 2_147_483_647:
                raise ValueError("seed 超出范围")
            cache_payload = json.dumps(
                {"format": CACHE_FORMAT, "engine": "carla-controlspace-32",
                 "features": ["control_history", "topology"], "sample": sample_key,
                 "model": model_id, "seed": seed, "condition": custom_condition,
                 "obstacles": obstacles},
                sort_keys=True, separators=(",", ":"))
            cache_key = hashlib.sha1(cache_payload.encode()).hexdigest()[:12]
            cache_path = os.path.join(
                CACHE_DIR, "%s__%s__seed%d__%s.json"
                % (model_id, sample_key, seed, cache_key))
            started = time.perf_counter()
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as handle:
                    result = json.load(handle)
                result["cache_hit"] = True
            else:
                with inference_lock:
                    clear_generation_cache()
                    result = generate(sample_key, model_id, seed,
                                      custom_condition, obstacles)
                    with open(cache_path, "w", encoding="utf-8") as handle:
                        json.dump(result, handle, separators=(",", ":"))
            result["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000, 1)
            self.send_json(200, result)
        except Exception as error:
            self.send_json(400, {"error": str(error)})

    def log_message(self, fmt, *args):
        print("[dashboard-carla] %s" % (fmt % args), flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT", "8765"))
    print("CARLA Diffusion Lens API on http://localhost:%d (%s)" % (port, device),
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

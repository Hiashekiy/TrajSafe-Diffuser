"""Local GPU inference service for the CARLA Diffusion Lens dashboard.

Same HTTP contract as the legacy ``backend.py`` (GET /health, POST /generate),
but the engine is :mod:`engine_carla`:

    Q_t [1,32,2] --(fixed B-spline decode)--> P_t [1,128,2] --> network
    -> topology (argmax pi, then FROZEN after activation)
    -> fixed Skeleton centres c_i = Gamma_m(i/127)
    -> 128 convex regions -> overlap check -> point-seeded gap bridge
    -> frozen corridor + exact B-spline constraint pack
    -> per-reverse-step control-space ALM on the fresh Q0_raw
    -> DDIM with Q0_safe -> curve

``alm_enabled: false`` runs ablation A (raw diffusion).  The response carries
``x0_raw_history`` (pre-ALM), ``x0_history`` (post-ALM / DDIM input), the frozen
``corridor`` (network + bridge cells) and the per-step ALM statistics.

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
from urllib.parse import parse_qs, urlparse

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
PAYLOAD_FORMAT = 4
CACHE_FORMAT = PAYLOAD_FORMAT

# UI/cache contract string shared with the web client (``/health.engine`` and
# the cached payload signature).  It is deliberately NOT derived from the
# control count: it identifies the control-space backend, the actual C travels
# in the payload (``pack_summary.num_controls``) and comes from the config.
CONTROL_SPACE_ENGINE = "carla-controlspace-32"
CATALOG_PATH = os.path.join(SITE_ROOT, "lib",
                             "dashboard-catalog-carla-160k8.json")

# The panel browses the WHOLE processed dataset instead of a pre-baked list:
# the split is picked first (train/val/test), then the sample inside it, either
# randomly or by typing its index.  ``/splits`` and ``/sample`` serve that
# metadata on demand, so nothing has to be baked into the front-end bundle.
SPLITS = ("train", "val", "test")
RES = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
    catalog = json.load(handle)
occupancy_cache = {}


def normalize_split(split) -> str:
    name = str("test" if split is None else split).strip().lower()
    if name not in SPLITS:
        raise ValueError("未知数据划分 %r（可选：%s）"
                         % (split, ", ".join(SPLITS)))
    return name


def split_size(split: str) -> int:
    return int(len(get_engine().split_data(split)["conditions"]))


def parse_key(sample_key):
    """Sample key -> (split or None, index or None).

    "test_0018" -> ("test", 18);  "18" -> (None, 18);  "val" -> ("val", None).
    """
    text = str(sample_key).strip().lower()
    if "_" in text:
        head, _, tail = text.rpartition("_")
        if head in SPLITS and tail.isdigit():
            return head, int(tail)
    if text in SPLITS:
        return text, None
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return None, int(digits)
    raise ValueError("无法识别的样本 %r" % (sample_key,))


def resolve_sample(sample_key, split=None):
    """(sample_key, split) -> (split, index).

    The split chosen in the panel wins; otherwise it comes from the key prefix
    ("train_0100") and finally falls back to test, so a stale front-end bundle
    or a hand-typed key can never break the panel.
    """
    key_split, index = parse_key(sample_key)
    name = normalize_split(split) if split else (key_split or "test")
    if index is None:
        index = 0
    n = split_size(name)
    if not 0 <= index < n:
        raise ValueError("%s 的样本编号 %d 超出范围（0 … %d）"
                         % (name, index, n - 1))
    return name, index


def sample_occupancy(split: str, index: int) -> np.ndarray:
    key = (split, int(index))
    if key not in occupancy_cache:
        engine = get_engine()
        occupancy_cache[key] = engine.sample_arrays(split, int(index))[0]
        if len(occupancy_cache) > 64:
            occupancy_cache.pop(next(iter(occupancy_cache)))
    return occupancy_cache[key].copy()


def wall_runs(occupancy: np.ndarray):
    """Canonical occupancy [256,256] -> [[x, y_svg, width], ...] (SVG pixels)."""
    occ = np.asarray(occupancy) > 0.5
    runs = []
    for row in range(occ.shape[0]):
        y_svg = RES - 1 - row                  # scene y=-1 is the bottom row
        line = occ[row]
        x = 0
        while x < line.shape[0]:
            if not line[x]:
                x += 1
                continue
            start = x
            while x < line.shape[0] and line[x]:
                x += 1
            runs.append([int(start), int(y_svg), int(x - start)])
    return runs


def sample_metadata(split: str, index: int):
    """Everything the panel needs for ONE sample (occupancy is not included:
    it is drawn from the run-length map, and the sampler reads it directly)."""
    eng = get_engine()
    occupancy, condition, curve_gt = eng.sample_arrays(split, index)
    gt = {"P": np.round(curve_gt, 5).tolist()}
    control_gt = eng.split_control_gt(split, index)
    if control_gt is not None:
        gt["control"] = np.round(control_gt, 5).tolist()
    return {
        "key": "%s_%04d" % (split, int(index)),
        "split": split,
        "datasetId": int(index),
        "maze": "carla_%04d" % int(index),
        "condition": np.round(condition, 5).tolist(),
        "groundTruth": gt,
        "map": {"resolution": RES, "wallRuns": wall_runs(occupancy)},
    }


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
             obstacles=None, alm_enabled: bool | None = None, split=None):
    split, index = resolve_sample(sample_key, split)
    eng = get_engine()
    base = sample_occupancy(split, index)
    default_condition = eng.split_data(split)["conditions"][index]
    condition = np.asarray(
        custom_condition if custom_condition is not None else default_condition,
        dtype=np.float32)
    if condition.shape != (2, 2) or not np.isfinite(condition).all() \
            or np.abs(condition).max() > 1:
        raise ValueError("起终点必须是 [-1,1]² 内的两个坐标")
    occupancy = apply_obstacles(base, obstacles)
    payload = eng.generate(sample_key, split, index, occupancy, condition,
                           seed, model_id=model_id,
                           verify_regions=False,
                           alm_enabled=alm_enabled is not False)
    payload["split"] = split
    payload["obstacles"] = obstacles or []
    payload["format"] = PAYLOAD_FORMAT
    if alm_enabled is False:
        payload["alm_note"] = ("消融 A：raw diffusion（未建立安全走廊、"
                               "未做 ALM 修正）")
    else:
        payload["alm_note"] = ("WARMUP → TRY_ACTIVATE（128 凸区域 + overlap + "
                               "gap bridge，随后冻结）→ GUIDED（每个 reverse "
                               "step 做 control-space B-spline ALM，dual 跨步 "
                               "warm-start），DDIM 使用 Q0_safe")
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
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_json(200, {
                "status": "ready", "device": str(device),
                "engine": CONTROL_SPACE_ENGINE,
                "format": PAYLOAD_FORMAT,
                "features": ["control_history", "topology", "ellipse_history",
                             "corridor", "alm_stats", "raw_vs_safe",
                             "splits"],
                "checkpoints": sorted(CHECKPOINTS.keys()),
                "splits": {name: split_size(name) for name in SPLITS},
                "cached": len(os.listdir(CACHE_DIR))})
        elif parsed.path == "/splits":
            self.send_json(200, {
                "engine": CONTROL_SPACE_ENGINE, "format": PAYLOAD_FORMAT,
                "splits": [{"name": name, "count": split_size(name)}
                           for name in SPLITS],
                "provenance": catalog.get("provenance", {}),
                "checkpoints": sorted(CHECKPOINTS.keys())})
        elif parsed.path == "/sample":
            try:
                query = parse_qs(parsed.query)
                split = normalize_split(query.get("split", ["test"])[0])
                index = int(query.get("index", ["0"])[0])
                n = split_size(split)
                if not 0 <= index < n:
                    raise ValueError("%s 的样本编号 %d 超出范围（0 … %d）"
                                     % (split, index, n - 1))
                self.send_json(200, sample_metadata(split, index))
            except Exception as error:
                self.send_json(400, {"error": str(error)})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            sample_key = request.get("sample_key")
            split = request.get("split")
            model_id = request.get("model_id")
            seed = int(request.get("seed", 42))
            custom_condition = request.get("condition")
            obstacles = request.get("obstacles") or []
            alm_enabled = request.get("alm_enabled")
            resolved_split, resolved_index = resolve_sample(sample_key, split)
            sample_key = "%s_%04d" % (resolved_split, resolved_index)
            if model_id not in CHECKPOINTS:
                raise ValueError("未知模型")
            if seed < 0 or seed > 2_147_483_647:
                raise ValueError("seed 超出范围")
            cache_payload = json.dumps(
                {"format": CACHE_FORMAT, "engine": CONTROL_SPACE_ENGINE,
                 "features": ["control_history", "topology", "corridor",
                              "alm_stats", "raw_vs_safe"],
                 "sample": sample_key, "split": resolved_split,
                 "index": resolved_index,
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
                                      custom_condition, obstacles,
                                      alm_enabled=alm_enabled,
                                      split=resolved_split)
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

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

from engine_carla import (CHECKPOINTS, DATASETS, DEFAULT_DATASET, Engine,
                          dataset_catalog, dataset_key, dataset_root,
                          dataset_splits)

CACHE_DIR = os.path.join(SITE_ROOT, "cache-carla")
os.makedirs(CACHE_DIR, exist_ok=True)
# Bump whenever the payload schema changes: it is part of the cache key, so a
# restarted backend can never replay a payload produced by older code.
# 5: the cache key gained ``alm_enabled`` and ``steps`` (toggling the ALM
#    switch or the step count used to replay a payload generated with the OTHER
#    setting - that is the "缓存不会清理" bug), and the payload reports the
#    executed schedule (``steps`` / ``times``).
# 6: the panel can display SEVERAL processed caches (``dataset``), so the cache
#    key gained ``dataset``: without it, switching the dataset replayed the
#    payload generated on the other map (same bug class as the ALM/steps one).
PAYLOAD_FORMAT = 6
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
# Several caches are served at once (``dataset``): 160k8p (the campaign cache)
# and 160k4p (the stricter, TEST-ONLY k=4 cache), see ``engine_carla.DATASETS``.
SPLITS = ("train", "val", "test")
RES = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
    catalog = json.load(handle)
occupancy_cache = {}


def normalize_dataset(dataset) -> str:
    return dataset_key(dataset)


def dataset_split_names(dataset=None):
    return dataset_splits(dataset)[1]


def normalize_split(split, dataset=None) -> str:
    """Validate a split against the SPLITS THAT EXIST in that dataset.

    ``160k4p`` ships ``test`` only, so offering train/val for it would only
    produce a traceback in ``np.load``; raise a readable error instead.
    """
    name = str("test" if split is None else split).strip().lower()
    allowed = dataset_split_names(dataset)
    if name not in allowed:
        raise ValueError("数据集 %s 没有 %r 划分（可选：%s）"
                         % (normalize_dataset(dataset), name,
                            ", ".join(allowed)))
    return name


def split_size(split: str, dataset=None) -> int:
    return int(len(get_engine().split_data(split, dataset)["conditions"]))


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


def resolve_sample(sample_key, split=None, dataset=None):
    """(sample_key, split) -> (split, index).

    The split chosen in the panel wins; otherwise it comes from the key prefix
    ("train_0100") and finally falls back to test, so a stale front-end bundle
    or a hand-typed key can never break the panel.
    """
    dkey = normalize_dataset(dataset)
    key_split, index = parse_key(sample_key)
    if key_split is not None and key_split not in dataset_split_names(dkey):
        key_split = None
    name = normalize_split(split, dkey) if split else (key_split or "test")
    if index is None:
        index = 0
    n = split_size(name, dkey)
    if not 0 <= index < n:
        raise ValueError("%s/%s 的样本编号 %d 超出范围（0 … %d）"
                         % (dkey, name, index, n - 1))
    return name, index


def sample_occupancy(split: str, index: int, dataset=None) -> np.ndarray:
    dkey = normalize_dataset(dataset)
    key = (dkey, split, int(index))
    if key not in occupancy_cache:
        engine = get_engine()
        occupancy_cache[key] = engine.sample_arrays(split, int(index), dkey)[0]
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


def sample_metadata(split: str, index: int, dataset=None):
    """Everything the panel needs for ONE sample (occupancy is not included:
    it is drawn from the run-length map, and the sampler reads it directly)."""
    dkey = normalize_dataset(dataset)
    eng = get_engine()
    occupancy, condition, curve_gt = eng.sample_arrays(split, index, dkey)
    gt = {"P": np.round(curve_gt, 5).tolist()}
    control_gt = eng.split_control_gt(split, index, dkey)
    if control_gt is not None:
        gt["control"] = np.round(control_gt, 5).tolist()
    return {
        "key": "%s_%04d" % (split, int(index)),
        "split": split,
        "dataset": dkey,
        "datasetLabel": DATASETS[dkey]["label"],
        "datasetRoot": dataset_root(dkey)[1],
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
             obstacles=None, alm_enabled: bool | None = None, split=None,
             steps: int | None = None, times=None, dataset=None):
    dkey = normalize_dataset(dataset)
    split, index = resolve_sample(sample_key, split, dkey)
    eng = get_engine()
    base = sample_occupancy(split, index, dkey)
    default_condition = eng.split_data(split, dkey)["conditions"][index]
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
                           alm_enabled=alm_enabled is not False,
                           steps=steps, times=times, dataset=dkey)
    payload["split"] = split
    payload["dataset"] = dkey
    payload["datasetLabel"] = DATASETS[dkey]["label"]
    payload["datasetRoot"] = dataset_root(dkey)[1]
    payload["datasetNote"] = DATASETS[dkey].get("note", "")
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
                             "splits", "datasets"],
                "checkpoints": sorted(CHECKPOINTS.keys()),
                "dataset": DEFAULT_DATASET,
                "datasets": [d["id"] for d in dataset_catalog()],
                "splits": {name: split_size(name)
                           for name in dataset_split_names(DEFAULT_DATASET)},
                "cached": len(os.listdir(CACHE_DIR))})
        elif parsed.path == "/splits":
            query = parse_qs(parsed.query)
            requested = query.get("dataset", [None])[0]
            try:
                dkey = normalize_dataset(requested)
            except Exception as error:
                return self.send_json(400, {"error": str(error)})
            names = dataset_split_names(dkey)
            self.send_json(200, {
                "engine": CONTROL_SPACE_ENGINE, "format": PAYLOAD_FORMAT,
                "dataset": dkey,
                "datasetLabel": DATASETS[dkey]["label"],
                "datasetRoot": dataset_root(dkey)[1],
                # flat list for older bundles (the default dataset)
                "splits": [{"name": name, "count": split_size(name, dkey)}
                           for name in names],
                # every cache the panel may switch to, with its own counts
                "datasets": [
                    dict(entry,
                         counts={name: split_size(name, entry["id"])
                                 for name in entry["splits"]})
                    for entry in dataset_catalog()],
                "provenance": catalog.get("provenance", {}),
                "checkpoints": sorted(CHECKPOINTS.keys())})
        elif parsed.path == "/sample":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", [None])[0]
                dkey = normalize_dataset(dataset)
                split = normalize_split(query.get("split", ["test"])[0], dkey)
                index = int(query.get("index", ["0"])[0])
                n = split_size(split, dkey)
                if not 0 <= index < n:
                    raise ValueError("%s/%s 的样本编号 %d 超出范围（0 … %d）"
                                     % (dkey, split, index, n - 1))
                self.send_json(200, sample_metadata(split, index, dkey))
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
            dataset = request.get("dataset")
            model_id = request.get("model_id")
            seed = int(request.get("seed", 42))
            custom_condition = request.get("condition")
            obstacles = request.get("obstacles") or []
            alm_enabled = request.get("alm_enabled")
            steps = request.get("steps")
            if steps is not None:
                steps = int(steps)
                if not 1 <= steps <= 64:
                    raise ValueError("steps 必须在 1..64 之间")
            dkey = normalize_dataset(dataset)
            resolved_split, resolved_index = resolve_sample(sample_key, split,
                                                            dkey)
            sample_key = "%s_%04d" % (resolved_split, resolved_index)
            if model_id not in CHECKPOINTS:
                raise ValueError("未知模型")
            if seed < 0 or seed > 2_147_483_647:
                raise ValueError("seed 超出范围")
            cache_payload = json.dumps(
                {"format": CACHE_FORMAT, "engine": CONTROL_SPACE_ENGINE,
                 "features": ["control_history", "topology", "corridor",
                              "alm_stats", "raw_vs_safe", "datasets"],
                 "sample": sample_key, "split": resolved_split,
                 "index": resolved_index,
                 # the same sample/seed on the OTHER cache is a different map
                 # and must not replay this payload
                 "dataset": dkey,
                 "model": model_id, "seed": seed, "condition": custom_condition,
                 "obstacles": obstacles,
                 # WITHOUT these two every toggle of the ALM switch / step count
                 # replayed the cached payload of the OTHER setting
                 "alm_enabled": (None if alm_enabled is None
                                 else bool(alm_enabled)),
                 "steps": steps},
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
                                      split=resolved_split, steps=steps,
                                      dataset=dkey)
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

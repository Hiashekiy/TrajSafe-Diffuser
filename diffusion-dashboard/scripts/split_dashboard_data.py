"""Split the one-time batch export into a light catalog and request caches."""
import json
import os

import numpy as np

from sample_selection import MAZES, SAMPLE_IDS


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
source = os.path.join(BASE, "lib", "dashboard-data.json")
cache_dir = os.path.join(BASE, "cache")
os.makedirs(cache_dir, exist_ok=True)

catalog_path = os.path.join(BASE, "lib", "dashboard-catalog.json")
if os.path.exists(source):
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)
else:
    with open(catalog_path, "r", encoding="utf-8") as handle:
        existing_catalog = json.load(handle)
    data = {
        "provenance": existing_catalog["provenance"],
        "maps": existing_catalog["maps"],
        "samples": [],
    }

dataset_dir = os.path.join(BASE, "..", "data", "processed_scene_v1", "test")
positions = np.load(os.path.join(dataset_dir, "positions.npy"))
ellipses = np.load(os.path.join(dataset_dir, "ellipses6.npy"))
conditions = np.load(os.path.join(dataset_dir, "conditions.npy"))
maze_ids = np.load(os.path.join(dataset_dir, "maze_id.npy"))


def rounded(array: np.ndarray, decimals: int = 5):
    return np.round(array.astype(np.float64), decimals).tolist()

catalog = {
    "provenance": data["provenance"],
    "maps": data["maps"],
    "samples": [],
}
for sample in data["samples"]:
    for model_id, history in sample["models"].items():
        cached = {
            "sampleKey": sample["key"], "modelId": model_id, "seed": 42,
            "cacheHit": True, "stateLabels": data["stateLabels"],
            "schedule": data["schedule"], **history,
        }
        path = os.path.join(cache_dir, f"{model_id}__{sample['key']}__seed42.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(cached, handle, separators=(",", ":"))

for dataset_id in SAMPLE_IDS:
    maze = MAZES[int(maze_ids[dataset_id])]
    catalog["samples"].append({
        "key": f"{maze}-{dataset_id}",
        "maze": maze,
        "datasetId": dataset_id,
        "condition": rounded(conditions[dataset_id]),
        "groundTruth": {
            "P": rounded(positions[dataset_id]),
            "E6": rounded(ellipses[dataset_id]),
        },
    })

with open(catalog_path, "w", encoding="utf-8") as handle:
    json.dump(catalog, handle, separators=(",", ":"))

print("catalog and request caches written")

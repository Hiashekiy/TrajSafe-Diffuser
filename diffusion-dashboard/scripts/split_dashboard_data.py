"""Split the one-time batch export into a light catalog and request caches."""
import json
import os


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
source = os.path.join(BASE, "lib", "dashboard-data.json")
cache_dir = os.path.join(BASE, "cache")
os.makedirs(cache_dir, exist_ok=True)

with open(source, "r", encoding="utf-8") as handle:
    data = json.load(handle)

catalog = {
    "provenance": data["provenance"],
    "maps": data["maps"],
    "samples": [],
}
for sample in data["samples"]:
    catalog["samples"].append({key: sample[key] for key in ("key", "maze", "datasetId", "condition", "groundTruth")})
    for model_id, history in sample["models"].items():
        cached = {
            "sampleKey": sample["key"], "modelId": model_id, "seed": 42,
            "cacheHit": True, "stateLabels": data["stateLabels"],
            "schedule": data["schedule"], **history,
        }
        path = os.path.join(cache_dir, f"{model_id}__{sample['key']}__seed42.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(cached, handle, separators=(",", ":"))

with open(os.path.join(BASE, "lib", "dashboard-catalog.json"), "w", encoding="utf-8") as handle:
    json.dump(catalog, handle, separators=(",", ":"))

print("catalog and request caches written")

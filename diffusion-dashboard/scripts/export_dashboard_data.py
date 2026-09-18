"""Run real checkpoints and export their complete reverse-diffusion histories."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.diffusion.schedule import NoiseSchedule
from src.models.joint import JointPlanner
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


MODEL_SOURCES = {
    "epoch100": "outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced/epoch_100.pt",
    "continue100": "outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced_continue100/best.pt",
    "continue200": "outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced_continue200/best.pt",
    "center_safe_isolated": "outputs/ckpt_v1_center_safe_isolated/best.pt",
}
SAMPLE_IDS = [9, 18, 368, 376, 668, 676, 739, 855, 460]
MAZES = ("umaze", "medium", "large")


def rounded(array: np.ndarray, decimals: int = 5):
    return np.round(array.astype(np.float64), decimals).tolist()


def wall_runs(occupancy: np.ndarray):
    runs = []
    h, w = occupancy.shape
    for source_y in range(h):
        row = occupancy[source_y] > 0.5
        x = 0
        while x < w:
            if not row[x]:
                x += 1
                continue
            start = x
            while x < w and row[x]:
                x += 1
            runs.append([start, h - 1 - source_y, x - start])
    return runs


@torch.no_grad()
def sample_history(model, schedule, cond, maps, seed=42):
    """Exact full-step sampler, recording model input x_t and final x_0 output."""
    torch.manual_seed(seed)
    batch, horizon = cond.shape[0], model.horizon
    start, goal = cond[:, 0], cond[:, 1]
    p = torch.randn(batch, horizon, 2, device=cond.device)
    e = torch.randn(batch, horizon, 6, device=cond.device)
    p[:, 0], p[:, -1] = start, goal
    sqrt_ab = schedule.sqrt_alphas_cumprod.detach().cpu().tolist()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod.detach().cpu().tolist()
    p_history, e_history, labels = [], [], []
    model.eval()

    for t in reversed(range(schedule.num_timesteps)):
        p_history.append(p.detach().cpu().numpy())
        e_history.append(e.detach().cpu().numpy())
        labels.append(f"t={t}")
        tb = torch.full((batch,), t, device=cond.device, dtype=torch.long)
        ab = torch.full((batch,), float(sqrt_ab[t]), device=cond.device)
        out = model(p, e, maps, cond, tb, ab)
        x0_p, x0_e = out["x0_p"], out["x0_e"]
        x0_p[:, 0], x0_p[:, -1] = start, goal
        if t == 0:
            p, e = x0_p, x0_e
        else:
            eps_p = (p - float(sqrt_ab[t]) * x0_p) / float(sqrt_1ma[t])
            eps_e = (e - float(sqrt_ab[t]) * x0_e) / float(sqrt_1ma[t])
            p = float(sqrt_ab[t - 1]) * x0_p + float(sqrt_1ma[t - 1]) * eps_p
            e = float(sqrt_ab[t - 1]) * x0_e + float(sqrt_1ma[t - 1]) * eps_e
        p[:, 0], p[:, -1] = start, goal

    p_history.append(p.detach().cpu().numpy())
    e_history.append(e.detach().cpu().numpy())
    labels.append("x0")
    return np.stack(p_history, axis=1), np.stack(e_history, axis=1), labels


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "config_v1_continue.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = os.path.join(ROOT, "data", "processed_scene_v1", "test")
    positions = np.load(os.path.join(test_dir, "positions.npy"))
    ellipses = np.load(os.path.join(test_dir, "ellipses6.npy"))
    conditions = np.load(os.path.join(test_dir, "conditions.npy"))
    maze_ids = np.load(os.path.join(test_dir, "maze_id.npy"))

    map_arrays = {name: np.load(os.path.join(ROOT, "data", "processed_scene_v1", "maps", f"{name}.npy")) for name in MAZES}
    cond = torch.as_tensor(conditions[SAMPLE_IDS], dtype=torch.float32, device=device)
    maps = torch.stack([torch.as_tensor(map_arrays[MAZES[int(maze_ids[index])]], dtype=torch.float32) for index in SAMPLE_IDS], dim=0).unsqueeze(1).to(device)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"]).to(device)

    histories = {}
    labels = []
    for model_id, relative_checkpoint in MODEL_SOURCES.items():
        print(f"sampling {model_id} on {device}", flush=True)
        model = JointPlanner(cfg["model"]).to(device)
        load_checkpoint(os.path.join(ROOT, relative_checkpoint), model, map_location=device)
        p_hist, e_hist, labels = sample_history(model, schedule, cond, maps)
        histories[model_id] = (p_hist, e_hist)
        del model

    result = {
        "provenance": {
            "dataset": "data/processed_scene_v1/test",
            "checkpoints": MODEL_SOURCES,
            "sceneUnits": "[-1,1]^2",
            "horizon": 128,
            "timesteps": 16,
            "recording": "exact sampler input states at t=15..0 plus final x0",
        },
        "stateLabels": labels,
        "schedule": {
            "sqrtAlphaBar": rounded(schedule.sqrt_alphas_cumprod.cpu().numpy(), 8),
            "sqrtOneMinusAlphaBar": rounded(schedule.sqrt_one_minus_alphas_cumprod.cpu().numpy(), 8),
        },
        "maps": {name: {"resolution": 256, "wallRuns": wall_runs(array)} for name, array in map_arrays.items()},
        "samples": [],
    }
    for batch_index, dataset_id in enumerate(SAMPLE_IDS):
        maze = MAZES[int(maze_ids[dataset_id])]
        record = {
            "key": f"{maze}-{dataset_id}", "maze": maze, "datasetId": dataset_id,
            "condition": rounded(conditions[dataset_id]),
            "groundTruth": {"P": rounded(positions[dataset_id]), "E6": rounded(ellipses[dataset_id])},
            "models": {},
        }
        for model_id, (p_hist, e_hist) in histories.items():
            record["models"][model_id] = {"PHistory": rounded(p_hist[batch_index]), "E6History": rounded(e_hist[batch_index])}
        result["samples"].append(record)

    out_path = os.path.join(os.path.dirname(__file__), "..", "lib", "dashboard-data.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
    print(out_path, os.path.getsize(out_path))


if __name__ == "__main__":
    main()

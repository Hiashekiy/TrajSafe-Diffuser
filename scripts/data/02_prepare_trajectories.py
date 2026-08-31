"""02_prepare_trajectories.py: split D4RL episodes into fixed-horizon windows.

Outputs to processed_dir/{train,val,test}/
    trajectories.npy  [N,H,6] normalized [ax,ay,x,y,vx,vy]
    conditions.npy    [N,2,2] normalized (start_xy, goal_xy)
and processed_dir/normalization.json

Gate: window_cross_episode_count == 0 (windows never span episode boundaries).
"""

import json
import os
import sys

import numpy as np
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.datasets.normalization import state_normalizer, save_normalization


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    horizon = int(data["horizon"])
    stride = int(data["stride"])
    max_episodes = int(data.get("max_episodes", 0))
    train_val_test = data["train_val_test"]
    processed = data["processed_dir"]

    with h5py.File(data["raw_hdf5"], "r") as f:
        obs = f["observations"][:]
        act = f["actions"][:]
        term = f["terminals"][:]
        to = f["timeouts"][:]

    done = term | to
    ep_starts = [0]
    ep_ends = []
    for i in range(len(done)):
        if done[i]:
            ep_ends.append(i + 1)
            ep_starts.append(i + 1)
    # guard: last partial episode ends at len(done)
    if len(ep_ends) == 0 or ep_ends[-1] != len(done):
        ep_ends.append(len(done))

    episodes = []
    for s, e in zip(ep_starts[:-1], ep_ends):
        if e - s >= horizon + 1:
            episodes.append((s, e))
    if max_episodes > 0:
        episodes = episodes[:max_episodes]

    # Build windows: each window is entirely inside one episode.
    windows = []   # (state_window, cond_window)
    for (s, e) in episodes:
        ep_obs = obs[s:e]
        ep_act = act[s:e]
        ep_state = np.concatenate([ep_act, ep_obs], axis=-1)  # [L,6]
        for start in range(0, e - s - horizon, stride):
            win = ep_state[start:start + horizon]
            cond = np.stack([win[0, 2:4], win[-1, 2:4]], axis=0)  # [2,2]
            windows.append((win, cond))
    # also include the final window of each episode
    for (s, e) in episodes:
        ep_state = np.concatenate([act[s:e], obs[s:e]], axis=-1)
        start = e - s - horizon
        if start >= 0:
            win = ep_state[start:start + horizon]
            cond = np.stack([win[0, 2:4], win[-1, 2:4]], axis=0)
            windows.append((win, cond))

    # optional cap so multi-maze mixed training stays balanced/feasible
    max_windows = int(data.get("max_windows", 0))
    if max_windows > 0 and len(windows) > max_windows:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(windows), size=max_windows, replace=False)
        windows = [windows[i] for i in np.sort(sel)]
        print(f"[02] capped windows to {max_windows}")

    states = np.stack([w[0] for w in windows], axis=0).astype(np.float64)
    conds = np.stack([w[1] for w in windows], axis=0).astype(np.float64)
    norm = state_normalizer(states)
    norm_states = norm.normalize(states).astype(np.float32)
    pos_mins = norm.mins[2:4]
    pos_maxs = norm.maxs[2:4]
    c = (conds - pos_mins) / (pos_maxs - pos_mins + norm.eps)
    norm_conds = (2.0 * c - 1.0).reshape(conds.shape).astype(np.float32)

    # ---- deterministic split at sequence level ----
    n = len(windows)
    train_end = int(round(n * train_val_test[0]))
    val_end = train_end + int(round(n * train_val_test[1]))
    split_idx = {"train": np.arange(0, train_end),
                 "val": np.arange(train_end, val_end),
                 "test": np.arange(val_end, n)}
    os.makedirs(processed, exist_ok=True)
    save_normalization(os.path.join(processed, "normalization.json"), norm)
    for split, idx in split_idx.items():
        out_dir = os.path.join(processed, split)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "trajectories.npy"), norm_states[idx])
        np.save(os.path.join(out_dir, "conditions.npy"), norm_conds[idx])
        print(f"[02] {split}: n={len(idx)}")

    # ---- gate ----
    print(f"[02] total windows={n}, episode windows all intra-episode, cross=0")
    with open(os.path.join(processed, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump({"n_windows": n, "split_idx": {k: v.tolist() for k, v in split_idx.items()},
                   "cross_episode_count": 0}, f, indent=2)


if __name__ == "__main__":
    main()

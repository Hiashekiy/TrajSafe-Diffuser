"""Checkpoint save/load utilities."""

import os

import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, cfg=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"model_state": model.state_dict()}
    if optimizer is not None:
        data["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch
    if cfg is not None:
        data["config"] = cfg
    if extra:
        data.update(extra)
    torch.save(data, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    data = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])
    return data

"""Endpoint inpainting conditioning, matching the archived Janner Diffuser / RGG
reference: start/goal observations are hard-set at t=0 and t=H-1 during both
training and sampling.  The denoiser itself stays unconditional."""

import torch


def apply_endpoint_condition(x, cond, horizon, obs_start=2, obs_end=4):
    """Overwrite (x,y) -- and optionally velocity -- at the trajectory endpoints.

    x:     [B,H,6] normalized
    cond:  [B,2,obs_end-obs_start] = (start_obs, goal_obs)
    horizon: trajectory length H
    """
    x = x.clone()
    x[:, 0, obs_start:obs_end] = cond[:, 0, :]
    x[:, horizon - 1, obs_start:obs_end] = cond[:, 1, :]
    return x

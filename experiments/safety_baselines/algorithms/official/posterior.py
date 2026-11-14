"""Posterior coefficients and the level ladder, as the official code builds them.

* `posterior_coefficients`  Constrained Diffusers / SafeDiffuser `diffusion.py:117-127`
  and `:117-131` (identical arithmetic in both releases: `posterior_mean_coef1/2`,
  `posterior_variance`, `posterior_log_variance_clipped`).
* `level_ladder`  Constrained Diffusers `p_sample_loop` (`umaze-large/diffusion.py:503-510`):

      for i in reversed(range(-200, self.n_timesteps)):
          if i <= 0:  i_ = 1
          elif i > self.n_timesteps:  i_ = self.n_timesteps - 1
          else:  i_ = i

  i.e. the last T+200 steps revisit t=1, and `nonzero_mask = (1 - (t == 1))`
  (`:479`) makes every one of those revisits **noiseless**. Our earlier ladder put the
  extra steps at t=0 with their own noise draw, which is not the same thing.
"""
import numpy as np


def posterior_coefficients(ab, betas):
    ab = np.asarray(ab, dtype=float); betas = np.asarray(betas, dtype=float)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    var = betas * (1.0 - ab_prev) / (1.0 - ab)
    coef1 = betas * np.sqrt(ab_prev) / (1.0 - ab)
    coef2 = (1.0 - ab_prev) * np.sqrt(1.0 - betas) / (1.0 - ab)
    return coef1, coef2, var


def level_ladder(num_timesteps, extra_steps):
    """t index of every reverse step, including the `extra_steps` revisits of t=1.

    The official loop is `for i in reversed(range(-200, self.n_timesteps))`, i.e. it walks
    **down** from T-1 to 1 and then revisits t=1 (`-1 ... -200`). Building the order
    without the `reversed` runs the chain backwards and the sample diverges.
    """
    ladder = []
    for i in reversed(range(-int(extra_steps), int(num_timesteps))):
        if i <= 0:
            ladder.append(1)
        elif i > num_timesteps:
            ladder.append(num_timesteps - 1)
        else:
            ladder.append(i)
    return ladder


def is_noiseless(t):
    """`nonzero_mask = (1 - (t == 1).float())` (umaze-large/diffusion.py:479)."""
    return int(t) == 1

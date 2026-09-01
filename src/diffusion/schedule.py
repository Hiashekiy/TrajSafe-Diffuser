"""DDPM noise schedule for the trajectory diffusion model."""

import numpy as np
import torch


def betas_for_alpha_bar(num_timesteps, alpha_bar_fn, max_beta=0.999):
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def squaredcos_cap_v2(t):
    return float(np.cos((t + 0.008) / 1.008 * np.pi / 2.0) ** 2)


def linear_beta_schedule(num_timesteps, start=0.0001, end=0.02):
    return np.linspace(start, end, num_timesteps, dtype=np.float64)


class NoiseSchedule:
    def __init__(self, num_timesteps, beta_schedule="squaredcos_cap_v2",
                 beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = int(num_timesteps)
        self.beta_schedule = beta_schedule
        if beta_schedule == "squaredcos_cap_v2":
            betas = betas_for_alpha_bar(self.num_timesteps, squaredcos_cap_v2)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(self.num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"unknown beta_schedule {beta_schedule}")
        self.betas = torch.as_tensor(betas, dtype=torch.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones_like(self.alphas_cumprod[:1]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # posterior variance for reverse diffusion
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

    def to(self, device):
        for name in ["betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "posterior_variance"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0, t, noise=None):
        """x0 [B,H,6] -> x_t [B,H,6]."""
        x0 = x0.to(torch.float64) if x0.dtype != torch.float64 else x0
        if noise is None:
            noise = torch.randn_like(x0)
        t = t.long()
        sqrt_ab = self.sqrt_alphas_cumprod[t]  # [B]
        sqrt_one_ma = self.sqrt_one_minus_alphas_cumprod[t]
        x_t = sqrt_ab[:, None, None] * x0 + sqrt_one_ma[:, None, None] * noise
        return x_t

    def q_sample_zero_sum(self, z0, t, noise=None):
        """Forward diffusion on the zero-sum residual Z [B,N,2].

        Noise is projected onto the zero-sum subspace along the N axis so the
        diffusion state always satisfies sum_k z_k = 0.
        """
        z0 = z0.float()
        if noise is None:
            noise = torch.randn_like(z0)
            noise = noise - noise.mean(dim=1, keepdim=True)  # project over N
        t = t.long()
        sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None].to(z0.dtype)
        sqrt_one_ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None].to(z0.dtype)
        return (sqrt_ab * z0 + sqrt_one_ma * noise).float()

    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return (x_t - sqrt_one_ma * eps) / sqrt_ab.clamp(min=1e-8)

    def predict_eps_from_x0(self, x_t, t, x0):
        sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return (x_t - sqrt_ab * x0) / sqrt_one_ma.clamp(min=1e-8)

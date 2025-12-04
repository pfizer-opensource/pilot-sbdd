from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import CondOTSBridgeScheduler, Scheduler, SineCosineScheduler


class ContinuousFlow(nn.Module):

    def __init__(self, d: int, scheduler: Scheduler | None = None):
        super().__init__()
        self.d = d
        if scheduler is None:
            self.scheduler = CondOTSBridgeScheduler()
        else:
            self.scheduler = scheduler

    def sample_prior(self, n: int, device: str = "cpu"):
        return torch.randn(n, self.d, device=device)

    def forward_interpolate(
        self,
        x0: Tensor,
        x1: Tensor,
        t: float,
        scaled_eps: Optional[Tensor] = None,
    ):
        s = self.scheduler(t)
        xt = s.alpha_t * x1 + s.sigma_t * x0
        if scaled_eps is not None:
            xt = xt + scaled_eps
        return xt

    def get_velocity_from_data_pred(self, t: Tensor, xt: Tensor, x1_pred: Tensor):
        coeff = self.scheduler(t)
        denom = coeff.sigma_t
        if not isinstance(self.scheduler, SineCosineScheduler):
            denom = denom.clamp(min=1e-4)
        c = coeff.dalpha_t / denom
        c.clamp(max=(100.0))
        vel = (x1_pred - xt) * c
        return vel

    def reverse_sample(
        self,
        xt: Tensor,
        x1_pred: Tensor,
        t: Tensor,
        dt: float,
        noise: int,
        scale: float = 1.0,
        dt_pow: float = 0.5,
    ):
        assert noise in [0, 1]
        vel = self.get_velocity_from_data_pred(t, xt, x1_pred)
        xt = xt + vel * dt
        if noise:
            xt = xt + scale * torch.randn_like(xt) * (dt**dt_pow)
        return xt

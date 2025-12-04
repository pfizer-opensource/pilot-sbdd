import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import CondOTScheduler, Scheduler


class CategoricalFlow(nn.Module):
    def __init__(
        self,
        kind: str,
        num_classes: int,
        prior_distribution: Tensor,
        scheduler: Scheduler | None = None,
    ):
        super().__init__()
        self.kind = kind
        self.num_classes = num_classes
        if scheduler is None:
            self.scheduler = CondOTScheduler()
        else:
            self.scheduler = scheduler
        self.register_buffer("prior_distribution", prior_distribution)

    def sample_prior(
        self, size: Tuple[int], empirical: bool = False, device: str = "cpu"
    ):
        if empirical:
            reshape = False
            if isinstance(size, tuple):
                size_int = math.prod(size)
                reshape = True
            samples = torch.multinomial(
                self.prior_distribution, size_int, replacement=True
            ).squeeze()
            if reshape:
                samples = samples.view(*size)
        else:
            samples = torch.randint(
                low=0, high=self.num_classes, size=size, device=device
            )
        samples = F.one_hot(samples, num_classes=self.num_classes).float()
        return samples

    def forward_interpolate(self, x0: Tensor, x1: Tensor, t: float):
        # xt | (x0, x1)
        if x0.ndim == 1 and x0.dtype == torch.int64:
            x0 = F.one_hot(x0, num_classes=self.num_classes).float()
        if x1.ndim == 1 and x1.dtype == torch.int64:
            x1 = F.one_hot(x1, num_classes=self.num_classes).float()

        s = self.scheduler(t)
        xt = s.alpha_t * x1 + s.sigma_t * x0
        xt = torch.multinomial(xt, 1, replacement=True).squeeze()
        xt = F.one_hot(xt, num_classes=self.num_classes).float()
        return xt

    def _reverse_cmtc_sample(
        self,
        xt: Tensor,
        x1_pred: Tensor,
        t: Tensor,
        noise: int,
        dt: float,
    ):
        # Discrete Flow according to Campbell et al. (2024)
        # https://arxiv.org/abs/2402.04997
        k = self.num_classes
        k_1_t = torch.gather(x1_pred, -1, xt)
        s = self.scheduler(t)
        # kappa_coeff = 1 / (1 - t) # for conditional OT
        kappa_coeff = s.dalpha_t / s.sigma_t.clamp(min=1e-4)
        forward_vel = dt * (1 + noise + noise * (k - 1) * t) * kappa_coeff * x1_pred
        backward_vel = dt * noise * k_1_t
        u_vel = (forward_vel + backward_vel).clamp(max=1.0)
        u_vel = u_vel.scatter(-1, xt, 0.0)
        pt = u_vel.scatter(
            -1,
            xt,
            (1.0 - u_vel.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        return pt

    def _reverse_flow_sample(
        self,
        xt: Tensor,
        x1_pred: Tensor,
        x0: Tensor,
        t: Tensor,
        noise: int,
        dt: float,
        alpha: float = 2.0,
        beta: float = 1.0,
    ):
        s = self.scheduler(t)
        # Discrete Flow according to Gat et al. (2024)
        # https://arxiv.org/abs/2407.15595
        # f_coeff = 1 / (1 - t) # for conditional OT
        f_coeff = s.dalpha_t / s.sigma_t.clamp(min=1e-4)

        dirac_xt = F.one_hot(xt.squeeze(), num_classes=self.num_classes).float()
        forward_vel = f_coeff * (x1_pred - dirac_xt)

        if noise:
            # b_coeff = 1 / t # for conditional OT
            b_coeff = s.dalpha_t / s.alpha_t.clamp(min=1e-4)
            backward_vel = b_coeff * (dirac_xt - x0)
        else:
            backward_vel = 0.0
            alpha = 1.0
            beta = 0.0

        u_vel = alpha * forward_vel - beta * backward_vel
        pt = (dirac_xt + dt * u_vel).clamp(min=0.0)
        return pt

    def reverse_sample(
        self,
        xt: Tensor,
        x1_pred: Tensor,
        x0: Tensor,
        t: Tensor,
        noise: int,
        dt: float,
        mode: str = "cmtc",
    ):
        # x_{t+1} | x_t
        assert noise in [0, 1]
        assert xt.ndim == 2 and xt.dtype == torch.int64

        if mode == "cmtc":
            pt = self._reverse_cmtc_sample(xt, x1_pred, t, noise, dt)
        else:
            pt = self._reverse_flow_sample(
                xt, x1_pred, x0, t, noise, dt, alpha=2.0, beta=1.0
            )

        xt = torch.multinomial(pt, 1, replacement=True).squeeze()
        xt = F.one_hot(xt, self.num_classes).float()
        return xt

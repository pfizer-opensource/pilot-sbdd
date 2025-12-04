from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from torch import Tensor, pi


class SchedulerOutput:
    def __init__(
        self,
        alpha_t: torch.Tensor,
        dalpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
        dsigma_t: torch.Tensor,
        gamma_t: torch.Tensor | None = None,
        dgamma_t: torch.Tensor | None = None,
    ):

        # for x1 (data)
        self.alpha_t = alpha_t
        self.dalpha_t = dalpha_t

        # for x0 (prior)
        self.sigma_t = sigma_t
        self.dsigma_t = dsigma_t

        # for intermediate (noise)
        self.gamma_t = gamma_t
        self.dgamma_t = dgamma_t

        # alpha(t) * x1 + beta(t) * x0 + gamma(t).sqrt() * z


class Scheduler(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> SchedulerOutput: ...

    def get_loss_weights(
        self, t: torch.Tensor, weighting: bool = True
    ) -> Tensor | float:
        if weighting:
            s = self(t)
            w = s.alpha_t / s.sigma_t.clamp(min=1e-3)
            w = torch.clamp(w, min=0.05, max=1.50).squeeze(-1)
        else:
            w = 1.0
        return w


class CondOTScheduler(Scheduler):
    """
    CondOT Scheduler.
    x_t = t * x_1 + (1 - t) * x_0
    """

    def __call__(self, t: torch.Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            dalpha_t=torch.ones_like(t),
            sigma_t=1.0 - t,
            dsigma_t=-torch.ones_like(t),
            gamma_t=None,
            dgamma_t=None,
        )


class CondOTSBridgeScheduler(Scheduler):
    """
    CondOT Schrödinger Brige Scheduler.
    x_t = t * x_1 + (1 - t) * x_0 + 1 * t * (1 - t) * z
    """

    def __call__(self, t: torch.Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            dalpha_t=torch.ones_like(t),
            sigma_t=1.0 - t,
            dsigma_t=-torch.ones_like(t),
            gamma_t=1.0 * t * (1.0 - t),
            dgamma_t=1.0 - 2.0 * t,
        )


class CosineNuScheduler(Scheduler):
    """
    Implementation as in Eq. 7 in https://arxiv.org/pdf/2102.09672 for s = 0.0
    Note we invert here the time variable t -> 1-t to be consistent with the diffusion timing
    """

    def __init__(self, nu: int = 1, latent_noise: bool = False):
        self.nu = nu
        self.latent_noise = latent_noise

    @classmethod
    def get_scheduler_coefficients(cls, t: Tensor, nu: int = 1):
        # signal
        y = 0.5 * pi * ((1 - t) ** nu)
        y = torch.cos(y)
        alpha_t = y**2
        # noise
        sigma_t = 1.0 - alpha_t
        return alpha_t, sigma_t

    def __call__(self, t: Tensor) -> SchedulerOutput:

        alpha_t, sigma_t = self.get_scheduler_coefficients(t, self.nu)

        input = 0.5 * pi * ((1 - t) ** self.nu)
        y = torch.cos(input)
        dalpha_t = (
            2
            * y
            * (-1.0)
            * torch.sin(input)
            * 0.5
            * pi
            * self.nu
            * (1 - t) ** (self.nu - 1)
            * (-1.0)
        )

        dsigma_t = -dalpha_t

        if self.latent_noise:
            gamma_t = t * (1.0 - t)
            dgamma_t = 1.0 - 2.0 * t
        else:
            gamma_t = None
            dgamma_t = None

        return SchedulerOutput(
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            sigma_t=sigma_t,
            dsigma_t=dsigma_t,
            gamma_t=gamma_t,
            dgamma_t=dgamma_t,
        )


class SineCosineScheduler(Scheduler):
    """_summary_
    As in https://arxiv.org/abs/2209.15571 in Eq 5
    """

    def __init__(self, latent_noise: bool = False):
        self.latent_noise = latent_noise

    def __call__(self, t: Tensor) -> SchedulerOutput:

        alpha_t = torch.sin(0.5 * pi * t)
        sigma_t = torch.cos(0.5 * pi * t)
        dalpha_t = pi / 2 * torch.cos(pi / 2 * t)
        dsigma_t = -pi / 2 * torch.sin(pi / 2 * t)

        if self.latent_noise:
            gamma_t = t * (1.0 - t)
            dgamma_t = 1.0 - 2.0 * t
        else:
            gamma_t = None
            dgamma_t = None

        return SchedulerOutput(
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            sigma_t=sigma_t,
            dsigma_t=dsigma_t,
            gamma_t=gamma_t,
            dgamma_t=dgamma_t,
        )


class SineConvex(Scheduler):
    def __init__(self, latent_noise: bool = False):
        self.latent_noise = latent_noise

    def __call__(self, t: Tensor) -> SchedulerOutput:

        alpha_t = torch.sin(0.5 * pi * t)
        sigma_t = 1.0 - alpha_t
        dalpha_t = 0.5 * pi * torch.cos(0.5 * pi * t)
        dsigma_t = -0.5 * pi * torch.sin(0.5 * pi * t)

        if self.latent_noise:
            gamma_t = t * (1.0 - t)
            dgamma_t = 1.0 - 2.0 * t
        else:
            gamma_t = None
            dgamma_t = None

        return SchedulerOutput(
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            sigma_t=sigma_t,
            dsigma_t=dsigma_t,
            gamma_t=gamma_t,
            dgamma_t=dgamma_t,
        )


class PolynomialConvexScheduler(Scheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            dalpha_t=self.n * (t ** (self.n - 1)),
            dsigma_t=-self.n * (t ** (self.n - 1)),
        )

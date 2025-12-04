import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from e3mol.experiments.utils import zero_mean


def nonzero_wrapper(log_a):
    ids = []
    first_nonzero = log_a[log_a != 0.0].max()
    for i, a in enumerate(log_a):
        if a == 0.0:
            ids.append(i)
    for i in ids:
        if i != 0:
            log_a[i] = first_nonzero / np.exp(1 / i)
    return log_a


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2,
    this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2


def get_beta_schedule(
    num_diffusion_timesteps: int = 1000,
    kind: str = "cosine",
    nu: float = 1.0,
    clamp_alpha_min=0.05,
    **kwargs,
):
    assert kind in ["cosine", "adaptive"]
    if kind == "cosine":
        s = kwargs.get("s")
        if s is None:
            s = 0.008
        steps = num_diffusion_timesteps + 2
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5)
            ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod = torch.from_numpy(
            clip_noise_schedule(alphas_cumprod, clip_value=clamp_alpha_min)
        )
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = alphas.clip(min=0.001)
        betas = 1 - alphas
        betas = torch.clip(betas, 0.0, 0.999).float()
    elif kind == "adaptive":
        s = kwargs.get("s")
        if s is None:
            s = 0.008
        steps = num_diffusion_timesteps + 2
        x = np.linspace(0, steps, steps)
        x = np.expand_dims(x, 0)  # ((1, steps))

        nu_arr = np.array(nu)  # (components, )  # X, charges, E, y, pos
        _steps = steps
        alphas_cumprod = (
            np.cos(0.5 * np.pi * (((x / _steps) ** nu_arr) + s) / (1 + s)) ** 2
        )  # ((components, steps))
        alphas_cumprod_new = alphas_cumprod / alphas_cumprod[:, 0]
        alphas_cumprod_new = clip_noise_schedule(
            alphas_cumprod_new.squeeze(), clip_value=clamp_alpha_min
        )[None, ...]
        # remove the first element of alphas_cumprod and then multiply
        # every element by the one before it
        alphas = alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1]
        # alphas[:, alphas.shape[1]-1] = 0.001
        alphas = alphas.clip(min=0.001)
        betas = 1 - alphas  # ((components, steps)) # X, charges, E, y, pos
        betas = np.swapaxes(betas, 0, 1)
        betas = torch.clip(torch.from_numpy(betas), 0.0, 0.999).squeeze().float()
    else:
        raise NotImplementedError
    return betas


class DiscreteDDPM(nn.Module):
    def __init__(
        self,
        scaled_reverse_posterior_sigma: bool = True,
        schedule: str = "cosine",
        nu: float = 1.0,
        enforce_zero_terminal_snr: bool = False,
        T: int = 500,
        clamp_alpha_min=0.05,
    ):
        super().__init__()
        self.scaled_reverse_posterior_sigma = scaled_reverse_posterior_sigma
        assert schedule in [
            "cosine",
            "adaptive",
        ]

        self.schedule = schedule
        self.T = T

        discrete_betas = get_beta_schedule(
            num_diffusion_timesteps=self.T,
            kind=self.schedule,
            nu=nu,
            alpha_clamp=clamp_alpha_min,
        )

        if enforce_zero_terminal_snr:
            discrete_betas = self.enforce_zero_terminal_snr_fnc(betas=discrete_betas)

        sqrt_betas = torch.sqrt(discrete_betas)
        alphas = 1.0 - discrete_betas
        sqrt_alphas = torch.sqrt(alphas)

        if schedule == "adaptive":
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            log_alpha_bar = nonzero_wrapper(log_alpha_bar)
            alphas_cumprod = torch.exp(log_alpha_bar)
            self._alphas = alphas
            self._log_alpha_bar = log_alpha_bar
            self._alphas_bar = torch.exp(log_alpha_bar)
            self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
            self._sigma_bar = torch.sqrt(self._sigma2_bar)
            self._gamma = (
                torch.log(-torch.special.expm1(2 * log_alpha_bar)) - 2 * log_alpha_bar
            )
        else:
            alphas_cumprod = torch.cumprod(alphas, dim=0)

        alphas_cumprod_prev = torch.nn.functional.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_1m_alphas_cumprod = sqrt_1m_alphas_cumprod.clamp(min=1e-4)

        if scaled_reverse_posterior_sigma:
            rev_variance = (
                discrete_betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            rev_variance[0] = rev_variance[1] / 2.0
            reverse_posterior_sigma = torch.sqrt(rev_variance)
        else:
            reverse_posterior_sigma = torch.sqrt(discrete_betas)

        self.register_buffer("discrete_betas", discrete_betas)
        self.register_buffer("sqrt_betas", sqrt_betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", sqrt_alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_1m_alphas_cumprod", sqrt_1m_alphas_cumprod)
        self.register_buffer("reverse_posterior_sigma", reverse_posterior_sigma)

    def enforce_zero_terminal_snr_fnc(self, betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas

        return betas

    def marginal_prob(self, x: Tensor, t: Tensor, cumulative: bool = True):
        """_summary_
        Eq. 4 in https://arxiv.org/abs/2006.11239
        Args:
            x (Tensor): _description_ Continuous data feature tensor
            t (Tensor): _description_ Discrete time variable between 1 and T
            cumulative (bool): _description_ Whether to use cumulative or non-cumulative alphas
        Returns:
            _type_: _description_
        """
        assert x.size(0) == t.size(
            0
        ), "x and t must have the same size along the 0-th dimension"

        assert str(t.dtype) == "torch.int64"
        expand_axis = len(x.size()) - 1

        if cumulative:
            if self.schedule == "adaptive":
                signal = self.get_alpha_bar(t_int=t)
                std = self.get_sigma_bar(t_int=t)
            else:
                signal = self.sqrt_alphas_cumprod[t]
                std = self.sqrt_1m_alphas_cumprod[t]
        else:
            if self.schedule == "adaptive":
                gamma_t, gamma_s = self.get_gamma(t_int=t), self.get_gamma(t_int=t - 1)
                (
                    _,
                    sigma_t_given_s,
                    alpha_t_given_s,
                ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
                signal = alpha_t_given_s
                std = sigma_t_given_s
            else:
                raise NotImplementedError

        for _ in range(expand_axis):
            signal = signal.unsqueeze(-1)
            std = std.unsqueeze(-1)

        mean = signal * x

        return mean, std

    def get_alpha_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        a = self._alphas_bar.to(t_int.device)[t_int.long()]
        return a.float()

    def get_sigma_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma_bar.to(t_int.device)[t_int]
        return s.float()

    def get_sigma2_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma2_bar.to(t_int.device)[t_int]
        return s.float()

    def get_gamma(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        g = self._gamma.to(t_int.device)[t_int]
        return g.float()

    def get_alpha(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        a = self.alphas.to(t_int.device)[t_int.long()]
        return a.float()

    def sigma_pos_ts_sq(self, t_int, s_int):
        gamma_s = self.get_gamma(t_int=s_int)
        gamma_t = self.get_gamma(t_int=t_int)
        delta_soft = F.softplus(gamma_s) - F.softplus(gamma_t)
        sigma_squared = -torch.expm1(delta_soft)
        return sigma_squared

    def get_alpha_pos_ts(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        ratio = torch.exp(log_a_bar[t_int] - log_a_bar[s_int])
        return ratio.float()

    def get_alpha_pos_ts_sq(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        ratio = torch.exp(2 * log_a_bar[t_int] - 2 * log_a_bar[s_int])
        return ratio.float()

    def get_sigma_pos_sq_ratio(self, s_int, t_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        s2_s = -torch.expm1(2 * log_a_bar[s_int])
        s2_t = -torch.expm1(2 * log_a_bar[t_int])
        ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))
        return ratio.float()

    def get_x_pos_prefactor(self, s_int, t_int):
        """a_s (s_t^2 - a_t_s^2 s_s^2) / s_t^2"""
        a_s = self.get_alpha_bar(t_int=s_int)
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        prefactor = a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)
        return prefactor.float()

    # from EDM
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))
        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sigma(self, gamma):
        """Computes sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Computes alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def _optimal_transport_alignment(self, a: Tensor, b: Tensor):
        C = torch.cdist(a, b, p=2)
        _, dest_ind = linear_sum_assignment(C.cpu().numpy(), maximize=False)
        dest_ind = torch.tensor(dest_ind, device=a.device)
        b_sorted = b[dest_ind]
        return b_sorted

    def optimal_transport_alignment(
        self,
        pos_ligand: Tensor,
        pos_random: Tensor,
        batch: Tensor,
    ):
        # Performs earth-mover distance optimal transport alignment
        # between batch of two point clouds
        pos_ligand_splits = pos_ligand.split(batch.bincount().tolist(), dim=0)
        pos_random_splits = pos_random.split(batch.bincount().tolist(), dim=0)

        pos_random_updated = [
            self._optimal_transport_alignment(a, b)
            for a, b in zip(pos_ligand_splits, pos_random_splits)
        ]
        pos_random_updated = torch.cat(pos_random_updated, dim=0)
        return pos_random_updated

    def sample_pos(
        self,
        t,
        pos,
        data_batch,
        remove_mean=True,
        cumulative: bool = True,
        extend_t: bool = True,
        ot_alignment: bool = False,
    ):
        bs = int(data_batch.max()) + 1
        if extend_t:
            # the time variable is on the batch level and shared among all nodes.
            # this is mostly the default in diffusion models
            assert t.size(0) == bs, "t and must have size of batch_size"
            t = t[data_batch]
        else:
            # the time variable is per node and not shared among all nodes
            # this means the noise level is independent for each node
            assert t.size(0) == data_batch.size(
                0
            ), "t and data_batch must have the same size"

        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        noise_coords_true = torch.randn_like(pos)
        if remove_mean:
            noise_coords_true = zero_mean(
                noise_coords_true, batch=data_batch, dim_size=bs, dim=0
            )

        if ot_alignment:
            noise_coords_true = self.optimal_transport_alignment(
                pos, noise_coords_true, data_batch
            )

        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.marginal_prob(x=pos, t=t, cumulative=cumulative)
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true

        return noise_coords_true, pos_perturbed

    def sample(
        self, t, feature, data_batch, cumulative: bool = True, extend_t: bool = True
    ):
        noise_coords_true = torch.randn_like(feature)
        if extend_t:
            # the time variable is on the batch level and shared among all nodes.
            # this is mostly the default in diffusion models
            t = t[data_batch]
        else:
            # the time variable is per node and not shared among all nodes
            # this means the noise level is independent for each node
            assert t.size(0) == data_batch.size(
                0
            ), "t and data_batch must have the same size"
        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.marginal_prob(
            x=feature, t=t, cumulative=cumulative
        )
        feature_perturbed = mean_coords + std_coords * noise_coords_true

        return noise_coords_true, feature_perturbed

    def sample_reverse(
        self,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_index_global=None,
        edge_attrs=None,
        eta_ddim: float = 1.0,
        extend_t: bool = True,
    ):

        if extend_t:
            # the time variable is on the batch level and shared among all nodes.
            # this is mostly the default in diffusion models
            t = t[batch]
        else:
            # the time variable is per node and not shared among all nodes
            # this means the noise level is independent for each node
            assert t.size(0) == batch.size(0), "t and batch must have the same size"

        std = self.reverse_posterior_sigma[t].unsqueeze(-1)
        noise = torch.randn_like(xt)

        if edge_index_global is not None:
            noise = torch.randn_like(edge_attrs)
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            noise = torch.randn_like(xt)
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)

        sigmast = self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
        sigmas2t = sigmast.pow(2)

        sqrt_alphas = self.sqrt_alphas[t].unsqueeze(-1)
        sqrt_1m_alphas_cumprod_prev = torch.sqrt(
            (1.0 - self.alphas_cumprod_prev[t]).clamp_min(1e-4)
        ).unsqueeze(-1)
        one_m_alphas_cumprod_prev = sqrt_1m_alphas_cumprod_prev.pow(2)
        sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev[t].unsqueeze(-1))
        one_m_alphas = self.discrete_betas[t].unsqueeze(-1)

        mean = (
            sqrt_alphas * one_m_alphas_cumprod_prev * xt
            + sqrt_alphas_cumprod_prev * one_m_alphas * model_out
        )
        mean = (1.0 / sigmas2t) * mean
        xt_m1 = mean + eta_ddim * std * noise

        if edge_index_global is None and cog_proj:
            xt_m1 = zero_mean(xt_m1, batch=batch, dim_size=bs, dim=0)

        return xt_m1

    def sample_reverse_adaptive(
        self,
        s,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_attrs=None,
        edge_index_global=None,
        eta_ddim: float = 1.0,
        probability_flow_ode=False,
        extend_t: bool = True,
    ):
        if edge_index_global is not None:
            noise = torch.randn_like(edge_attrs)
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            noise = torch.randn_like(xt)
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)

        sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s, t_int=t)
        prefactor1 = self.get_sigma2_bar(t_int=t)
        prefactor2 = self.get_sigma2_bar(t_int=s) * self.get_alpha_pos_ts_sq(
            t_int=t, s_int=s
        )
        sigma2_t_s = prefactor1 - prefactor2
        noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
        noise_prefactor = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)
        z_t_prefactor = (
            self.get_alpha_pos_ts(t_int=t, s_int=s) * sigma_sq_ratio
        ).unsqueeze(-1)
        x_prefactor = self.get_x_pos_prefactor(s_int=s, t_int=t).unsqueeze(-1)

        a = z_t_prefactor[batch] * xt if extend_t else z_t_prefactor * xt
        b = x_prefactor[batch] * model_out if extend_t else x_prefactor * model_out

        mu = a + b
        if probability_flow_ode:
            noise = torch.zeros_like(noise)
        xt_m1 = (
            mu + eta_ddim * noise_prefactor[batch] * noise
            if extend_t
            else mu + eta_ddim * noise_prefactor * noise
        )
        if edge_index_global is None and cog_proj:
            xt_m1 = zero_mean(xt_m1, batch=batch, dim_size=bs, dim=0)
        return xt_m1

    def probability_flow_ode(self, t, xt, model_out, batch, extend_t: bool = True):
        # x(t) = m(t)*x(0) + s(t)*z
        # z = x(t) - m(t)*x(0) / s(t)
        if extend_t:
            t = t[batch]

        betas = self.discrete_betas[t].unsqueeze(-1)
        m = self.alphas_cumprod[t].unsqueeze(-1)
        s = (1.0 - m).sqrt()

        score = (xt - m.sqrt() * model_out) / s
        xt = xt + 0.5 * betas * xt + 0.5 * betas * score

        return xt

    def snr_s_t_weighting(self, s, t, device, clamp_min=None, clamp_max=None):
        signal_s = self.alphas_cumprod[s]
        noise_s = 1.0 - signal_s
        snr_s = signal_s / noise_s

        signal_t = self.alphas_cumprod[t]
        noise_t = 1.0 - signal_t
        snr_t = signal_t / noise_t
        weights = snr_s - snr_t
        if clamp_min:
            weights = weights.clamp_min(clamp_min)
        if clamp_max:
            weights = weights.clamp_max(clamp_max)
        return weights.to(device)

    def snr_t_weighting(
        self, t, device, clamp_min: float = 0.05, clamp_max: float = 1.5
    ):
        weights = (
            (self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t]))
            .clamp(min=clamp_min, max=clamp_max)
            .to(device)
        )
        return weights

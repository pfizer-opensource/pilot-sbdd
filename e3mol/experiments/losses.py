from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_scatter import scatter_add, scatter_mean

from e3mol.experiments.data.datainfo import get_vdw_radius_from_integer_np


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        modalities: Optional[List] = None,
        param: Optional[List] = None,
    ) -> None:
        super().__init__()

        if modalities is None:
            modalities = ["coords", "atoms", "charges", "bonds"]
        if param is None:
            param = ["data", "data", "data", "data"]

        assert len(modalities) == len(param)
        self.modalities = modalities
        self.param = param

        if "coords" in modalities:
            self.regression_key = "coords"
        elif "latents" in modalities:
            self.regression_key = "latents"
        else:
            raise ValueError

    def loss_non_nans(self, loss: Tensor, modality: str) -> Tensor:
        m = loss.isnan()
        if torch.any(m):
            print(f"Recovered NaNs in {modality}. Selecting NoN-Nans")
        return loss[~m], m

    def aggregate_loss(
        self,
        loss: Tensor,
        weights: Tensor,
        batch_size: int,
        batch: Tensor,
        variable_mask: Optional[Tensor] = None,
        node_level_t: bool = False,
    ):

        if variable_mask is None:
            variable_mask = torch.ones_like(loss)

        loss = variable_mask * loss

        if node_level_t:
            loss = weights * loss
            loss = scatter_add(loss, index=batch, dim=0, dim_size=batch_size)
            loss, m = self.loss_non_nans(loss, self.regression_key)
            loss = torch.mean(loss)
        else:
            loss = scatter_mean(loss, index=batch, dim=0, dim_size=batch_size)
            loss, m = self.loss_non_nans(loss, self.regression_key)
            loss *= weights[~m]
            loss = torch.sum(loss, dim=0)
        return loss

    def forward(
        self,
        true_data: Dict,
        pred_data: Dict,
        batch: Tensor,
        bond_aggregation_index: Tensor,
        variable_mask: Tensor,
        weights: Tensor,
        intermediate_coords: bool = False,
        aux_weight: float = 1.0,
        l1_loss: bool = False,
        node_level_t: bool = False,
    ) -> Dict[str, Optional[Tensor]]:

        batch_size = len(batch.unique())

        if not node_level_t:
            assert len(weights) == batch_size
        else:
            assert len(weights) == true_data[self.regression_key].size(0)

        if intermediate_coords:
            pos_true = true_data[self.regression_key]
            pos_list = pred_data[self.regression_key]
            # tensor of shape [num_layers, N, 3]
            # where the last element on first axis is the final coords prediction
            pos_losses = (
                torch.nn.functional.l1_loss(
                    pos_true[None, :, :].expand(len(pos_list), -1, -1),
                    pos_list,
                    reduction="none",
                )
                if l1_loss
                else torch.square(pos_true.unsqueeze(0) - pos_list)
            )
            pos_losses = pos_losses.mean(-1)  # [num_layers, N]
            pos_losses = scatter_mean(pos_losses, batch, -1)
            # [num_layers, bs]
            aux_loss = pos_losses[:-1].mean(0)
            pos_loss = pos_losses[-1]
            regr_loss = pos_loss + aux_weight * aux_loss
            regr_loss, m = self.loss_non_nans(regr_loss, self.regression_key)
            regr_loss *= weights[~m]
            regr_loss = torch.sum(regr_loss, dim=0)
        else:
            regr_loss = F.mse_loss(
                pred_data[self.regression_key],
                true_data[self.regression_key],
                reduction="none",
            ).mean(-1)
            regr_loss = self.aggregate_loss(
                regr_loss, weights, batch_size, batch, variable_mask, node_level_t
            )

        if self.param[self.modalities.index("atoms")] == "data":
            fnc = F.cross_entropy
            take_mean = False
        else:
            fnc = F.mse_loss
            take_mean = True

        atoms_loss = fnc(pred_data["atoms"], true_data["atoms"], reduction="none")
        if take_mean:
            atoms_loss = atoms_loss.mean(dim=1)
        atoms_loss = self.aggregate_loss(
            atoms_loss, weights, batch_size, batch, variable_mask, node_level_t
        )

        if "charges" in self.modalities:
            if self.param[self.modalities.index("charges")] == "data":
                fnc = F.cross_entropy
                take_mean = False
            else:
                fnc = F.mse_loss
                take_mean = True

            charges_loss = fnc(
                pred_data["charges"], true_data["charges"], reduction="none"
            )
            if take_mean:
                charges_loss = charges_loss.mean(dim=1)
            charges_loss = self.aggregate_loss(
                charges_loss, weights, batch_size, batch, variable_mask, node_level_t
            )

        if "bonds" in self.modalities:
            if self.param[self.modalities.index("bonds")] == "data":
                fnc = F.cross_entropy
                take_mean = False
            else:
                fnc = F.mse_loss
                take_mean = True

            bonds_loss = fnc(pred_data["bonds"], true_data["bonds"], reduction="none")
            if take_mean:
                bonds_loss = bonds_loss.mean(dim=1)
            bonds_loss = 0.5 * scatter_mean(
                bonds_loss,
                index=bond_aggregation_index,
                dim=0,
                dim_size=true_data["atoms"].size(0),
            )
            bonds_loss = self.aggregate_loss(
                bonds_loss, weights, batch_size, batch, None, node_level_t
            )

        if "hybridization" in self.modalities:
            hybridization_loss = F.cross_entropy(
                pred_data["hybridization"],
                true_data["hybridization"],
                reduction="none",
            )
            hybridization_loss = self.aggregate_loss(
                hybridization_loss,
                weights,
                batch_size,
                batch,
                variable_mask,
                node_level_t,
            )
        else:
            hybridization_loss = None

        if "numHs" in self.modalities:
            numHs_loss = F.cross_entropy(
                pred_data["numHs"],
                true_data["numHs"],
                reduction="none",
            )
            numHs_loss = self.aggregate_loss(
                numHs_loss, weights, batch_size, batch, variable_mask, node_level_t
            )
        else:
            numHs_loss = None

        loss = {
            self.regression_key: regr_loss,
            "atoms": atoms_loss,
            "charges": charges_loss,
            "bonds": bonds_loss,
            "hybridization": hybridization_loss,
            "numHs": numHs_loss,
        }

        return loss


def mollifier_cutoff(input: torch.Tensor, cutoff: torch.Tensor, eps: torch.Tensor):
    r""" Mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff: Cutoff radius.
        eps: Offset added to distances for numerical stability.

    """
    mask = (input + eps < cutoff).float()
    exponent = 1.0 - 1.0 / (1.0 - torch.pow(input * mask / cutoff, 2))
    cutoffs = torch.exp(exponent)
    cutoffs = cutoffs * mask
    return cutoffs


def smoothe_morse_potential(r, rm, b=0.25, a=1.0):
    v = a * (1.0 - torch.exp(-b * (r - rm))).pow(2) * mollifier_cutoff(r, rm, 1e-4)
    return v


def ligand_pocket_clash_energy_loss(
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    x_ligand: Tensor,
    x_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    tolerance: float = 0.25,
    b: float = 0.25,
    a: float = 1.0,
    count: bool = False,
    min_distance: Optional[float] = None,
) -> Tensor:
    d = torch.sum(
        (pos_pocket.view(1, -1, 3) - pos_ligand.view(-1, 1, 3)) ** 2, dim=-1
    ).sqrt()  # (n_ligands, n_pockets)

    if min_distance is None:
        dm_ligand = get_vdw_radius_from_integer_np(x_ligand.detach().cpu().numpy())
        dm_pocket = get_vdw_radius_from_integer_np(x_pocket.detach().cpu().numpy())
        dm = dm_pocket[None, :] + dm_ligand[:, None] + tolerance
        dm = torch.from_numpy(dm).float().to(pos_ligand.device)
    else:
        dm = torch.ones_like(d) * min_distance

    connectivity_mask = batch_ligand.view(-1, 1) == batch_pocket.view(1, -1)
    if not count:
        energy = smoothe_morse_potential(d, rm=dm, b=b, a=a)
        energy = energy * connectivity_mask
    else:
        energy = (d <= dm) * connectivity_mask
    energy = energy.sum(dim=1)  # (n_ligands,)
    return energy


def shifted_sigmoid(k, x):
    assert (x >= 0).all()
    return torch.sigmoid(k - x)


def lddt_loss(
    pos_ligand_true: Tensor,
    pos_ligand_pred: Tensor,
    pos_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    cutoff: float = 10.0,
) -> Tensor:

    connectivity_mask = batch_ligand.view(-1, 1) == batch_pocket.view(1, -1)

    d_true = torch.sum(
        (pos_pocket.view(1, -1, 3) - pos_ligand_true.view(-1, 1, 3)) ** 2, dim=-1
    ).sqrt()  # (n_ligands, n_pockets)

    cutoff_mask = (d_true < cutoff).float() * connectivity_mask
    d_true = d_true * cutoff_mask

    d_pred = torch.sum(
        (pos_pocket.view(1, -1, 3) - pos_ligand_pred.view(-1, 1, 3)) ** 2, dim=-1
    ).sqrt()  # (n_ligands, n_pockets)
    d_pred = d_pred * connectivity_mask * cutoff_mask

    dist_l1 = (d_pred - d_true).abs()

    eps = (
        F.sigmoid(0.5 - dist_l1)
        + F.sigmoid(1.0 - dist_l1)
        + F.sigmoid(2.0 - dist_l1)
        + F.sigmoid(4.0 - dist_l1)
    ) / 4.0

    lddt = (cutoff_mask * eps).sum(-1) / cutoff_mask.sum(-1)  # (n_ligands,)
    lddt = 1.0 - lddt
    return lddt

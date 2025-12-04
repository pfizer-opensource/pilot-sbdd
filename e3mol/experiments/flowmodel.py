from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_add, scatter_mean, scatter_min
from tqdm import tqdm

import e3mol.experiments.guidance as gd
import e3mol.experiments.utils as ut
from e3mol.experiments.data.datainfo import DatasetInfo
from e3mol.experiments.data.molecule import Molecule
from e3mol.experiments.flow.base import (
    CondOTSBridgeScheduler,
    CondOTScheduler,
    CosineNuScheduler,
)
from e3mol.experiments.flow.categorical import CategoricalFlow
from e3mol.experiments.flow.continuous import ContinuousFlow
from e3mol.experiments.guidance import pocket_clash_guidance, pseudo_lj_guidance
from e3mol.experiments.losses import ligand_pocket_clash_energy_loss
from e3mol.experiments.utils import (
    clamp_norm,
    coalesce_edges,
    concat_ligand_pocket,
    get_edge_mask_inpainting,
    get_generated_molecules,
    get_joint_edge_attrs,
    get_masked_com,
    remove_mean_ligand,
    remove_mean_pocket,
)
from e3mol.modules.model import DenoisingEdgeNetwork, load_model_from_ckpt


def get_coords_std(n: torch.Tensor, clamp_max=4.0):
    std = torch.sqrt(0.23 * n).clamp_max(clamp_max)
    return std


def symmetrize_edge_attr(edge_index, edge_attr):
    j, i = edge_index
    mask = j < i
    mask_i = i[mask]
    mask_j = j[mask]
    edge_attr_triu = edge_attr[mask]
    j = torch.concat([mask_j, mask_i])
    i = torch.concat([mask_i, mask_j])
    edge_index = torch.stack([j, i], dim=0)
    edge_attr = torch.concat([edge_attr_triu, edge_attr_triu], dim=0)
    edge_index, edge_attr = sort_edge_index(
        edge_index=edge_index,
        edge_attr=edge_attr,
        sort_by_row=False,
    )
    return edge_index, edge_attr


def wrap_sample_edge_prior(cat_bonds, cat_prior, edge_index_ligand, n, device):
    edge_attr_ligand = cat_bonds.sample_prior(
        size=(n, n), empirical=cat_prior == "empirical", device=device
    )
    edge_attr_ligand = edge_attr_ligand[edge_index_ligand[0], edge_index_ligand[1], :]
    edge_index_ligand, edge_attr_ligand = symmetrize_edge_attr(
        edge_index_ligand, edge_attr_ligand
    )
    return edge_index_ligand, edge_attr_ligand


def get_diffusion_callable(noise_schedule: str = "constant-one") -> Callable:
    """
    Returns a callable function for the diffusion process based on the given noise schedule.

    Supported noise schedules:
      - "sine": returns sin^2(pi * t)
      - "brownian": returns t * (1 - t)
      - "constant-one": returns a tensor of ones with the same shape as t

    :param noise_schedule: The noise schedule identifier.
    :return: A function f(t) implementing the diffusion schedule.
    :raises ValueError: If an unsupported noise_schedule is provided.
    """
    if noise_schedule == "sine":
        return lambda t: torch.sin(t * torch.pi).pow(2)
    elif noise_schedule == "brownian":
        return lambda t: t * (1 - t)
    elif noise_schedule == "constant-one":
        return lambda t: torch.ones_like(t)
    else:
        raise ValueError(f"Unsupported noise_schedule: {noise_schedule}")


class EQGATDiffPL(nn.Module):
    def __init__(
        self,
        hparams: Dict[str, Any],
        dataset_info: DatasetInfo,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        if "context_pos_noise" not in hparams.keys():
            hparams["context_pos_noise"] = 0.0
        if "node_level_t" not in hparams.keys():
            hparams["node_level_t"] = False
        if "ot_alignment" not in hparams.keys():
            hparams["ot_alignment"] = False
        if "addNumHs" not in hparams.keys():
            hparams["addNumHs"] = False
        if "fragmentation_mix" not in hparams.keys():
            hparams["fragmentation_mix"] = False
        if "fragment_prior" not in hparams.keys():
            hparams["fragment_prior"] = "fragment"
        if "apply_kabsch" not in hparams.keys():
            hparams["apply_kabsch"] = False
        if "include_field_repr" not in hparams.keys():
            hparams["include_field_repr"] = False
        if "com_ligand" not in hparams.keys():
            hparams["com_ligand"] = False
        if "constrained_flow" not in hparams.keys():
            hparams["constrained_flow"] = False
        if "model_score" not in hparams.keys():
            hparams["model_score"] = False
        if "omit_intermediate_noise" not in hparams.keys():
            hparams["omit_intermediate_noise"] = False
        if "gamma2_func" in hparams.keys():
            hparams["gamma2_func"] = "t(1-t)"

        self.apply_kabsch = hparams["apply_kabsch"]

        self.cat_prior = "uniform"
        self.hparams = namedtuple("hparams", hparams.keys())(*hparams.values())
        self.dataset_info = dataset_info
        # raw input charges start from -2
        self.charge_offset = 2
        self.num_atom_types = hparams["num_atom_types"]
        self.num_bond_classes = hparams["num_bond_classes"]
        self.num_charge_classes = hparams["num_charge_classes"]
        self.num_atom_features = self.num_atom_types + self.num_charge_classes

        atom_types_prior = dataset_info.atom_types.float()
        charge_types_prior = dataset_info.charges_marginals.float()
        bond_types_prior = dataset_info.edge_types.float()
        self.register_buffer("atoms_prior", atom_types_prior.clone())
        self.register_buffer("bonds_prior", bond_types_prior.clone())
        self.register_buffer("charges_prior", charge_types_prior.clone())

        # Flows
        latent_noise = not self.hparams.omit_intermediate_noise
        pos_scheduler = CondOTSBridgeScheduler()
        # use cosine scheduler as default like in PILOT diffusion
        nu = 2.0
        pos_scheduler = CosineNuScheduler(nu=nu, latent_noise=latent_noise)
        print(pos_scheduler)
        self.position_flow = ContinuousFlow(
            d=3,
            scheduler=pos_scheduler,
        )
        self.cat_atoms = CategoricalFlow(
            kind="atoms",
            num_classes=self.num_atom_types,
            prior_distribution=self.atoms_prior.clone(),
            scheduler=CondOTScheduler(),
        )
        self.cat_charge = CategoricalFlow(
            kind="charges",
            num_classes=self.num_charge_classes,
            prior_distribution=self.charges_prior.clone(),
            scheduler=CondOTScheduler(),
        )
        self.cat_bonds = CategoricalFlow(
            kind="bonds",
            num_classes=self.num_bond_classes,
            prior_distribution=self.bonds_prior.clone(),
            scheduler=CondOTScheduler(),
        )

        self.select_node_order = ["atom_types", "charges"]
        if self.hparams.addNumHs:
            self.num_Hs = 5  # [0, 1, 2, 3, 4]
            self.num_atom_features += self.num_Hs
            self.select_node_order.append("numHs")
            numHs_prior = dataset_info.numHs.float()
            self.register_buffer("numHs_prior", numHs_prior.clone())
            self.cat_numHs = CategoricalFlow(
                kind="numHs",
                num_classes=self.num_Hs,
                prior_distribution=self.numHs_prior.clone(),
                scheduler=CondOTScheduler(),
            )

        if self.hparams.addHybridization:
            self.num_hybridization = 9  # [..., sp3, sp2, sp, ...]
            self.num_atom_features += self.num_hybridization
            self.select_node_order.append("hybridization")
            hybridization_prior = dataset_info.hybridization.float()
            self.register_buffer("hybridization_prior", hybridization_prior.clone())
            self.cat_hybridization = CategoricalFlow(
                kind="hybridization",
                num_classes=self.num_hybridization,
                prior_distribution=self.hybridization_prior.clone(),
                scheduler=CondOTScheduler(),
            )

        self.anchor_feature = int(self.hparams.fragmentation)
        if self.anchor_feature:
            self.fragment_embedding = nn.Embedding(
                num_embeddings=2,
                embedding_dim=hparams["sdim"],
                # padding_idx=0
            )
            self.anchor_embedding = nn.Embedding(
                num_embeddings=2,
                embedding_dim=hparams["sdim"],
                # padding_idx=0
            )
        else:
            self.fragment_embedding = None
            self.anchor_embedding = None

        if ckpt_path is None:
            self.model = DenoisingEdgeNetwork(
                num_atom_features=self.num_atom_features,
                num_bond_types=self.num_bond_classes,
                hn_dim=(hparams["sdim"], hparams["vdim"]),
                edge_dim=hparams["edim"],
                cutoff_local=hparams["cutoff_local"],
                num_layers=hparams["num_layers"],
                latent_dim=hparams["latent_dim"],
                atom_mapping=True,
                bond_mapping=True,
                use_out_norm=hparams["use_out_norm"],
                store_intermediate_coords=hparams["store_intermediate_coords"],
                joint_property_prediction=hparams["joint_property_prediction"],
                regression_property=hparams["regression_property"],
                node_level_t=hparams["node_level_t"],
                model_score=hparams["model_score"],
                include_field_repr=hparams["include_field_repr"],
            )
        else:
            print(f"Loading model from ckpt path {ckpt_path}")
            self.model = load_model_from_ckpt(ckpt_path, old=False)

        self.T = hparams["timesteps"]
        self.node_level_t = (
            hparams["node_level_t"] if "node_level_t" in hparams.keys() else False
        )

        self.cutoff_p = hparams["cutoff_local"]
        self.cutoff_lp = hparams["cutoff_local"]

    def forward(
        self,
        batch: Batch,
        t: Tensor,
        z: Optional[Tensor] = None,
        latent_gamma: float = 1.0,
        pocket_noise_std: float = 0.1,
        fragment_anchor_mask: Optional[Tensor] = None,
    ):

        atom_types_ligand: Tensor = batch.x
        atom_types_pocket: Tensor = batch.x_pocket
        pos_ligand: Tensor = batch.pos
        pos_pocket: Tensor = batch.pos_pocket
        charges_ligand: Tensor = batch.charges
        batch_ligand: Tensor = batch.batch
        batch_pocket: Tensor = batch.pos_pocket_batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )
        N_ligand = len(pos_ligand)

        device = pos_ligand.device

        pocket_noise = torch.randn_like(pos_pocket) * pocket_noise_std
        pos_pocket = pos_pocket + pocket_noise

        if not self.hparams.com_ligand:
            pos_centered_ligand, pos_centered_pocket, _ = remove_mean_pocket(
                pos_ligand, pos_pocket, batch_ligand, batch_pocket
            )  # pocket is 0-CoM and ligand is translated to (previous) pocket CoM
        else:
            pos_centered_ligand, pos_centered_pocket, _ = remove_mean_ligand(
                pos_ligand, pos_pocket, batch_ligand, batch_pocket
            )  # ligand is 0-CoM and protein is translated to (previous) ligand CoM

        # anchor point and variable fragment
        if fragment_anchor_mask is not None:
            fragment_mask_ligand = fragment_anchor_mask[:, 0]
            fragment_mask_pocket = torch.zeros(
                (pos_pocket.shape[0],), device=device, dtype=torch.long
            )  # 0 means fixed, 1 means variable
            fragment_mask = torch.cat(
                [fragment_mask_ligand, fragment_mask_pocket], dim=0
            )
            fragment_embedding = self.fragment_embedding(fragment_mask.long())

            anchor_mask = fragment_anchor_mask[:, 1]

            # ligand is shifted into by pocket com,
            # now we shift again on the reference (anchor or variable fragment) com
            if self.hparams.fragment_prior == "anchor":
                ref_mask = fragment_anchor_mask[
                    :, 1
                ]  # shifted prior to atom(s) where molecule is cut
            elif self.hparams.fragment_prior == "fragment":
                ref_mask = fragment_anchor_mask[
                    :, 0
                ]  # shifted prior to atoms where fragment is replaced

            ref_point = get_masked_com(
                pos_centered_ligand, ref_mask.float(), batch_ligand
            )
            nVariables = scatter_add(
                fragment_mask_ligand, batch_ligand, dim=0
            )  # (batch_size,)
            uncond_case = (
                nVariables.long() == batch_ligand.bincount().long()
            )  # (batch_size,)

            anchor_mask_pocket = torch.zeros(
                (pos_pocket.shape[0],), device=device, dtype=torch.long
            )
            anchor_mask = torch.cat([anchor_mask, anchor_mask_pocket], dim=0)
            anchor_embedding = self.anchor_embedding(anchor_mask.long())
            anchor_fragment_embedding = anchor_embedding + fragment_embedding
            fragment_mask = fragment_mask.unsqueeze(-1).float()
        else:
            anchor_fragment_embedding = None
            fragment_mask = None
            fragment_mask_ligand = torch.ones((len(pos_ligand),), device=device).bool()
            ref_point = None

        # ligand edges
        # fully-connected ligand
        edge_index_global_lig = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
        edge_index_global_lig = sort_edge_index(
            edge_index_global_lig, sort_by_row=False
        )
        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        # one-hot-encode discrete features
        atom_types_ligand = F.one_hot(atom_types_ligand, self.num_atom_types).float()
        charges_ligand = F.one_hot(
            charges_ligand + self.charge_offset, self.num_charge_classes
        ).float()
        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig, self.num_bond_classes
        ).float()

        # sample latent prior
        pos_prior = torch.randn_like(pos_centered_ligand)
        # OT
        if fragment_mask is not None:
            # shift com to reference points of the variable part in the inpainting case
            pos_prior = (
                pos_prior
                + ref_point[batch_ligand]
                * (~uncond_case)[batch_ligand].unsqueeze(-1).float()
            )
            fragment_mask_ids = torch.where(fragment_mask_ligand.squeeze().bool())[0]
            pos_prior_subset = pos_prior[fragment_mask_ids]
            pos_centered_ligand_subset = pos_centered_ligand[fragment_mask_ids]
            batch_ligand_subset = batch_ligand[fragment_mask_ids]
            pos_prior_subset = self.optimal_transport_alignment(
                pos_centered_ligand_subset,
                pos_prior_subset,
                batch_ligand_subset,
                apply_kabsch=self.apply_kabsch,
            )
            pos_prior[fragment_mask_ids] = pos_prior_subset
            pos_centered_ligand[fragment_mask_ids] = pos_centered_ligand_subset
        else:
            pos_prior = self.optimal_transport_alignment(
                pos_centered_ligand,
                pos_prior,
                batch_ligand,
                apply_kabsch=self.apply_kabsch,
            )

        # already one-hot-encoded
        atom_types_prior = self.cat_atoms.sample_prior(
            size=(N_ligand,), device=device, empirical=self.cat_prior == "empirical"
        )
        charges_prior = self.cat_charge.sample_prior(
            size=(N_ligand,), device=device, empirical=self.cat_prior == "empirical"
        )
        edges_prior = self.cat_bonds.sample_prior(
            size=(N_ligand, N_ligand),
            device=device,
            empirical=self.cat_prior == "empirical",
        )

        edges_prior = edges_prior[edge_index_global_lig[0], edge_index_global_lig[1], :]
        batch_edge_global = batch_ligand[edge_index_global_lig[0]]  #
        if self.node_level_t:
            _j, _i = edge_index_global_lig
            t_node = t
            assert t.size(0) == batch_ligand.size(
                0
            ), "t and data_batch must have the same size"
            t_ = t.squeeze()
            assert t_.ndim == 1, "t must be 1D"
            t_ = (t_.view(1, -1) + t_.view(-1, 1)) / 2.0  # (N, N)
            t_edge = t_[_j, _i].unsqueeze(-1)
        else:
            t_node = t[batch_ligand]
            t_edge = t[batch_edge_global]

        edge_index_global_lig, edges_prior = symmetrize_edge_attr(
            edge_index_global_lig, edges_prior
        )

        atom_types_ligand_perturbed = self.cat_atoms.forward_interpolate(
            x0=atom_types_prior, x1=atom_types_ligand, t=t_node
        )
        charges_ligand_perturbed = self.cat_charge.forward_interpolate(
            x0=charges_prior, x1=charges_ligand, t=t_node
        )
        edge_attr_global_perturbed_lig = self.cat_bonds.forward_interpolate(
            x0=edges_prior, x1=edge_attr_global_lig, t=t_edge
        )
        edge_index_global_lig, edge_attr_global_perturbed_lig = symmetrize_edge_attr(
            edge_index_global_lig, edge_attr_global_perturbed_lig
        )

        if not self.hparams.omit_intermediate_noise:
            sigma = self.position_flow.scheduler(t_node).gamma_t.sqrt()
            if sigma.ndim == 1:
                sigma = sigma.unsqueeze(-1)
            eps = torch.randn_like(pos_prior)
            scaled_eps = sigma * eps
        else:
            sigma = None
            eps = None
            scaled_eps = None

        # fragment-anchor-mask
        # 0: fixed, 1: variable
        if self.hparams.node_level_t and self.hparams.context_pos_noise > 0.0:
            v = fragment_mask_ligand.unsqueeze(-1)  # variable means 1
            noise_context = torch.randn_like(pos_prior) * self.hparams.context_pos_noise
            pos_prior = v * pos_prior + (1.0 - v) * (
                pos_centered_ligand + noise_context
            )
            t_pos_node = scatter_min(t_node, batch_ligand, dim=0)[0]
            t_pos_node = t_pos_node[batch_ligand]
        else:
            t_pos_node = t_node

        pos_perturbed_ligand = self.position_flow.forward_interpolate(
            x0=pos_prior,
            x1=pos_centered_ligand,
            t=t_pos_node,
            scaled_eps=scaled_eps,
        )

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand_perturbed,
            "charges": charges_ligand_perturbed,
            "pos": pos_perturbed_ligand,
            "batch": batch_ligand,
        }

        # protein pocket features (atom-types, charges) need to be one-hot encoded
        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charges_pocket = torch.zeros(
            pos_pocket.shape[0],
            charges_ligand_perturbed.shape[1],
            dtype=torch.float32,
            device=device,
        )

        pocket_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_pocket,
            "charges": charges_pocket,
            "pos": pos_centered_pocket,
            "batch": batch_pocket,
        }

        if self.hparams.addNumHs:
            numHs_ligand: Tensor = F.one_hot(batch.numHs, self.num_Hs).float()
            numHs_prior = self.cat_numHs.sample_prior(
                size=(N_ligand,), device=device, empirical=self.cat_prior == "empirical"
            )
            numHs_ligand_perturbed = self.cat_numHs.forward_interpolate(
                x0=numHs_prior, x1=numHs_ligand, t=t_node
            )
            numHs_pocket = torch.zeros(
                pos_pocket.shape[0],
                numHs_ligand_perturbed.shape[1],
                dtype=torch.float32,
                device=device,
            )
            ligand_feats["numHs"] = numHs_ligand_perturbed
            pocket_feats["numHs"] = numHs_pocket

        if self.hparams.addHybridization:
            hybridization_ligand: Tensor = F.one_hot(
                batch.hybridization, self.num_hybridization
            ).float()
            hybridization_prior = self.cat_hybridization.sample_prior(
                size=(N_ligand,), device=device, empirical=self.cat_prior == "empirical"
            )
            hybridization_ligand_perturbed = self.cat_hybridization.forward_interpolate(
                x0=hybridization_prior, x1=hybridization_ligand, t=t_node
            )
            hybridization_pocket = torch.zeros(
                pos_pocket.shape[0],
                hybridization_ligand_perturbed.shape[1],
                dtype=torch.float32,
                device=device,
            )
            ligand_feats["hybridization"] = hybridization_ligand_perturbed
            pocket_feats["hybridization"] = hybridization_pocket

        # Concatenate Ligand-Pocket node features
        pl_feats_dict = concat_ligand_pocket(
            ligand_feat_dict=ligand_feats, pocket_feat_dict=pocket_feats
        )

        # Concatenate all node features along the feature dims
        atom_feats_in_perturbed = torch.cat(
            [pl_feats_dict[feat] for feat in self.select_node_order],
            dim=-1,
        )
        pos_joint_perturbed = pl_feats_dict["pos"]
        batch_full = pl_feats_dict["batch"]
        ligand_mask = pl_feats_dict["ligand_mask"]

        # Create edge features
        (
            edge_index_global,
            edge_attr_global_perturbed,
            batch_edge_global,
            edge_mask_ligand,
            _,
            edge_initial_interaction,
        ) = get_joint_edge_attrs(
            pos_perturbed_ligand,
            pos_centered_pocket,
            batch_ligand,
            batch_pocket,
            edge_attr_global_perturbed_lig,
            self.num_bond_classes,
            cutoff_p=self.cutoff_p,
            cutoff_lp=self.cutoff_lp,
        )

        if self.node_level_t:
            assert t.size(0) == batch_ligand.size(
                0
            ), "timesteps must be equal to the number of nodes for ligands"
            tpocket = scatter_mean(t_pos_node, batch_ligand, dim=0)
            tpocket = tpocket[batch_pocket]
            t = torch.concat([t_pos_node, tpocket], dim=0)

        # forward pass for the model
        if fragment_mask is None:
            fragment_mask = ligand_mask.clone()
            fragment_mask = fragment_mask.unsqueeze(-1).float()

        # field representations
        if self.hparams.include_field_repr:
            v0 = pocket_clash_guidance(
                pos_ligand=pos_perturbed_ligand,
                pos_pocket=pos_centered_pocket,
                batch_ligand=batch_ligand,
                batch_pocket=batch_pocket,
            )[1].detach()
            v0 = clamp_norm(v0, maxnorm=10.0)
            v1 = pseudo_lj_guidance(
                pos_ligand=pos_perturbed_ligand,
                pos_pocket=pos_centered_pocket,
                batch_ligand=batch_ligand,
                batch_pocket=batch_pocket,
            )[1].detach()
            v1 = clamp_norm(v1, maxnorm=10.0)
            v_01 = torch.stack([v0, v1], dim=-1)
            v_ligand = torch.zeros((v_01.size(0), 3, self.hparams.sdim), device=device)
            v_ligand[:, :, :2] = v_01
            v_pocket = torch.zeros(
                (pos_pocket.size(0), 3, self.hparams.sdim), device=device
            )
            v = torch.cat([v_ligand, v_pocket], dim=0)
        else:
            v = None

        if self.hparams.context_pos_noise > 0.0:
            variable_mask = ligand_mask.float().unsqueeze(-1)
        else:
            variable_mask = fragment_mask

        out = self.model(
            x=atom_feats_in_perturbed,
            t=t,
            v=v,
            pos=pos_joint_perturbed,
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_perturbed,
            edge_index_ligand=edge_index_global_lig,
            batch=batch_full,
            edge_attr_initial_ohe=edge_initial_interaction,
            batch_edge_global=batch_edge_global,
            z=z,
            ligand_mask=ligand_mask,
            batch_ligand=batch_ligand,
            latent_gamma=latent_gamma,
            edge_mask_ligand=edge_mask_ligand,
            anchor_fragment_embedding=anchor_fragment_embedding,
            variable_mask=variable_mask,
        )
        out["t"] = t
        out["ligand_mask"] = ligand_mask

        if self.hparams.context_pos_noise > 0.0:
            out["variable_mask"] = torch.ones_like(fragment_mask_ligand)
        else:
            # loss computation only for variable fragment atoms
            out["variable_mask"] = fragment_mask_ligand

        # Ground truth masking
        out["coords_true"] = pos_centered_ligand
        out["atoms_true"] = atom_types_ligand.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_lig.argmax(dim=-1)
        out["charges_true"] = charges_ligand.argmax(dim=-1)
        out["bond_aggregation_index"] = edge_index_global_lig[1]
        out["eps"] = eps
        out["sigma"] = sigma

        out["coords_pocket"] = pos_centered_pocket
        out["atoms_pocket"] = atom_types_pocket.argmax(dim=-1)

        if self.hparams.addNumHs:
            out["numHs_true"] = numHs_ligand.argmax(dim=-1)
        else:
            out["numHs_true"] = None

        if self.hparams.addHybridization:
            out["hybridization_true"] = hybridization_ligand.argmax(dim=-1)
        else:
            out["hybridization_true"] = None
        return out

    def select_splitted_node_feats(
        self, x: Tensor, batch_num_nodes: Tensor, select: Tensor
    ):
        x_split = x.split(batch_num_nodes.cpu().numpy().tolist(), dim=0)
        x_select = torch.concat([x_split[i] for i in select.cpu().numpy()], dim=0)
        return x_select.to(x.device)

    def rigid_alignment(self, x_0, x_1, pre_centered=False):
        """
        See: https://en.wikipedia.org/wiki/Kabsch_algorithm
        Alignment of two point clouds using the Kabsch algorithm.
        Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
        """
        d = x_0.shape[1]
        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

        # remove COM from data and record initial COM
        if pre_centered:
            x_0_mean = torch.zeros(1, d)
            x_1_mean = torch.zeros(1, d)
            x_0_c = x_0
            x_1_c = x_1
        else:
            x_0_mean = x_0.mean(dim=0, keepdim=True)
            x_1_mean = x_1.mean(dim=0, keepdim=True)
            x_0_c = x_0 - x_0_mean
            x_1_c = x_1 - x_1_mean

        # Covariance matrix
        H = x_0_c.T.mm(x_1_c)
        U, S, V = torch.svd(H)
        # Rotation matrix
        R = V.mm(U.T)
        # Translation vector
        if pre_centered:
            t = torch.zeros(1, d)
        else:
            t = x_1_mean - R.mm(x_0_mean.T).T  # has shape (1, D)

        # apply rotation to x_0_c
        x_0_aligned = x_0_c.mm(R.T)

        # move x_0_aligned to its original frame
        x_0_aligned = x_0_aligned + x_0_mean

        # apply the translation
        x_0_aligned = x_0_aligned + t

        return x_0_aligned

    def _optimal_transport_alignment(self, a: Tensor, b: Tensor, apply_kabsch=False):
        C = torch.cdist(a, b, p=2)
        _, dest_ind = linear_sum_assignment(C.cpu().numpy(), maximize=False)
        dest_ind = torch.tensor(dest_ind, device=a.device)
        b_sorted = b[dest_ind]
        if apply_kabsch:
            b_sorted = self.rigid_alignment(b_sorted, a, pre_centered=False)

        return b_sorted

    def optimal_transport_alignment(
        self,
        pos_ligand: Tensor,
        pos_random: Tensor,
        batch: Tensor,
        apply_kabsch: bool,
    ):
        # Performs earth-mover distance optimal transport alignment
        # between batch of two point clouds
        pos_ligand_splits = pos_ligand.split(batch.bincount().tolist(), dim=0)
        pos_random_splits = pos_random.split(batch.bincount().tolist(), dim=0)

        pos_random_updated = [
            self._optimal_transport_alignment(a, b, apply_kabsch=apply_kabsch)
            for a, b in zip(pos_ligand_splits, pos_random_splits)
        ]
        pos_random_updated = torch.cat(pos_random_updated, dim=0)
        return pos_random_updated

    def reverse_sampling(
        self,
        N: int,
        num_graphs: int,
        pocket_data: Tensor,
        device: str,
        num_nodes_lig: Tensor,
        strategy: str = "linear",
        verbose: bool = False,
        sanitize=False,
        clash_guidance: bool = False,
        clash_guidance_start=None,
        clash_guidance_end=None,
        clash_guidance_scale: float = 0.1,
        eps: float = 1e-3,
        save_traj: bool = False,
        z: Optional[Tensor] = None,
        cat_noise: bool = True,
        pos_noise: bool = False,
        min_distance: float = 2.0,
        score_dynamics: bool = False,
        score_scale: float = 0.5,
        dt_pow: float = 0.5,
        corrector_last: int = 3,
        noise_schedule: str | Callable = "constant-one",
        discrete_gat: bool = False,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:
        """Implements unconditional ligand generation"""

        if isinstance(noise_schedule, str):
            assert noise_schedule in ["sine", "brownian", "constant-one"]
            noise_fnc = get_diffusion_callable(noise_schedule=noise_schedule)
        else:
            assert isinstance(noise_schedule, Callable)
            noise_fnc = noise_schedule

        if score_dynamics:
            assert self.hparams.model_score

        pos_pocket = pocket_data.pos_pocket.to(device)
        batch_pocket = pocket_data.pos_pocket_batch.to(device)
        x_pocket = pocket_data.x_pocket.to(device)

        batch_ligand = torch.arange(num_graphs, device=device).repeat_interleave(
            num_nodes_lig, dim=0
        )

        coords_traj = []
        atoms_traj = []
        edges_traj = []
        edges_pred_traj = []

        n = len(batch_ligand)
        # fully-connected edge-index for ligand
        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)

        pos_ligand_initial: Tensor = pocket_data.pos.to(device)
        batch_ligand_initial: Tensor = pocket_data.batch.to(device)
        if not self.hparams.com_ligand:
            pos_ligand, pos_pocket, com = remove_mean_pocket(
                pos_ligand_initial, pos_pocket, batch_ligand_initial, batch_pocket
            )
        else:
            pos_ligand, pos_pocket, com = remove_mean_ligand(
                pos_ligand_initial, pos_pocket, batch_ligand_initial, batch_pocket
            )
        # initialize the latent variables / features
        atom_types_ligand = self.cat_atoms.sample_prior(
            size=(n,), empirical=self.cat_prior == "empirical", device=device
        )
        charge_types_ligand = self.cat_charge.sample_prior(
            size=(n,), empirical=self.cat_prior == "empirical", device=device
        )
        edge_attr_ligand = self.cat_bonds.sample_prior(
            size=(n, n), empirical=self.cat_prior == "empirical", device=device
        )
        edge_attr_ligand = edge_attr_ligand[
            edge_index_ligand[0], edge_index_ligand[1], :
        ]
        edge_index_ligand, edge_attr_ligand = symmetrize_edge_attr(
            edge_index_ligand, edge_attr_ligand
        )

        pos_ligand = self.position_flow.sample_prior(n=n, device=device)

        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charge_types_pocket = torch.zeros(
            pos_pocket.shape[0],
            charge_types_ligand.shape[1],
            dtype=torch.float32,
            device=device,
        )

        if self.anchor_feature:
            anchor_mask_ligand = torch.zeros((n,), device=device, dtype=torch.long)
            anchor_mask_pocket = torch.zeros(
                (len(pos_pocket),), device=device, dtype=torch.long
            )
            anchor_mask = torch.cat([anchor_mask_ligand, anchor_mask_pocket], dim=0)
            anchor_embedding = self.anchor_embedding(anchor_mask)

            fragment_mask_ligand = torch.ones(
                (n,), device=device, dtype=torch.long
            )  # variable
            fragment_mask_pocket = torch.zeros(
                (len(pos_pocket),), device=device, dtype=torch.long
            )
            fragment_mask = torch.cat(
                [fragment_mask_ligand, fragment_mask_pocket], dim=0
            )
            fragment_embedding = self.fragment_embedding(fragment_mask)
            fragment_mask = fragment_mask.unsqueeze(-1).float()
            anchor_fragment_embedding = anchor_embedding + fragment_embedding
        else:
            fragment_mask = None
            anchor_fragment_embedding = None

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand,
            "charges": charge_types_ligand,
            "pos": pos_ligand,
            "batch": batch_ligand,
        }

        pocket_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_pocket,
            "charges": charge_types_pocket,
            "pos": pos_pocket,
            "batch": batch_pocket,
        }

        if self.hparams.addNumHs:
            numHs_ligand = self.cat_numHs.sample_prior(
                size=(n,), device=device, empirical=self.cat_prior == "empirical"
            )
            numHs_pocket = torch.zeros(
                pos_pocket.shape[0],
                numHs_ligand.shape[1],
                dtype=torch.float32,
                device=device,
            )
            ligand_feats["numHs"] = numHs_ligand
            pocket_feats["numHs"] = numHs_pocket
        else:
            numHs_ligand = None

        if self.hparams.addHybridization:
            hybridization_ligand = self.cat_hybridization.sample_prior(
                size=(n,), device=device, empirical=self.cat_prior == "empirical"
            )
            hybridization_pocket = torch.zeros(
                pos_pocket.shape[0],
                hybridization_ligand.shape[1],
                dtype=torch.float32,
                device=device,
            )
            ligand_feats["hybridization"] = hybridization_ligand
            pocket_feats["hybridization"] = hybridization_pocket
        else:
            hybridization_ligand = None

        edge_index_ligand, edge_attr_ligand = sort_edge_index(
            edge_index_ligand, edge_attr_ligand, sort_by_row=False
        )

        # combine and get PL-complex features on node- and edge-level
        (
            pl_feats_dict,
            edge_index_global,
            edge_attr_global,
            batch_edge_global,
            edge_mask_ligand,
            edge_initial_interaction,
        ) = ut.combine_protein_ligand_feats(
            ligand_feats,
            pocket_feats,
            edge_attr_ligand,
            self.num_bond_classes,
            self.cutoff_p,
            self.cutoff_lp,
        )

        if strategy == "linear":
            chain = np.linspace(0 + eps, 1.0 - eps, N)
        elif strategy == "log":
            chain = 1.0 - np.geomspace(0 + eps, 1.0, N)
            chain = np.flip(chain)

        chain = torch.from_numpy(chain.copy()).float().to(device)
        iterator = tqdm(range(N), total=N) if verbose else range(N)

        if fragment_mask is None:
            fragment_mask = pl_feats_dict["ligand_mask"].clone()
            if fragment_mask.ndim == 1:
                fragment_mask = fragment_mask.unsqueeze(-1).float()

        if clash_guidance_start is None:
            clash_guidance_start = 0
        if clash_guidance_end is None:
            clash_guidance_end = N

        # forward pass for the model
        dt = 1 / len(chain)

        for i, step in enumerate(iterator):

            if i == len(iterator) - 1:
                if not self.hparams.remove_hs:
                    j = 1
                else:
                    j = corrector_last
            else:
                j = 1

            for _ in range(j):

                t = chain[i]
                node_feats_in = torch.cat(
                    [pl_feats_dict[feat] for feat in self.select_node_order], dim=-1
                )
                t_emb = torch.zeros((len(node_feats_in), 1), device=device).fill_(
                    t.item()
                )
                t_node = torch.zeros((len(pos_ligand), 1), device=device).fill_(
                    t.item()
                )
                t_edge = torch.zeros((len(edge_attr_ligand), 1), device=device).fill_(
                    t.item()
                )

                # field representations
                if self.hparams.include_field_repr:
                    v0 = pocket_clash_guidance(
                        pos_ligand=pos_ligand,
                        pos_pocket=pos_pocket,
                        batch_ligand=batch_ligand,
                        batch_pocket=batch_pocket,
                    )[1].detach()
                    v0 = clamp_norm(v0, maxnorm=10.0)
                    v1 = pseudo_lj_guidance(
                        pos_ligand=pos_ligand,
                        pos_pocket=pos_pocket,
                        batch_ligand=batch_ligand,
                        batch_pocket=batch_pocket,
                    )[1].detach()
                    v1 = clamp_norm(v1, maxnorm=10.0)
                    v_01 = torch.stack([v0, v1], dim=-1)
                    v_ligand = torch.zeros(
                        (v_01.size(0), 3, self.hparams.sdim), device=device
                    )
                    v_ligand[:, :, :2] = v_01
                    v_pocket = torch.zeros(
                        (pos_pocket.size(0), 3, self.hparams.sdim), device=device
                    )
                    v = torch.cat([v_ligand, v_pocket], dim=0)
                else:
                    v = None

                out = self.model(
                    x=node_feats_in,
                    t=t_emb,
                    v=v,
                    pos=pl_feats_dict["pos"],
                    edge_index=edge_index_global,
                    edge_attr=edge_attr_global,
                    edge_index_ligand=edge_index_ligand,
                    batch=pl_feats_dict["batch"],
                    edge_attr_initial_ohe=edge_initial_interaction,
                    batch_edge_global=batch_edge_global,
                    z=z,
                    ligand_mask=pl_feats_dict["ligand_mask"],
                    batch_ligand=batch_ligand,
                    latent_gamma=1.0,
                    edge_mask_ligand=edge_mask_ligand,
                    anchor_fragment_embedding=anchor_fragment_embedding,
                    variable_mask=fragment_mask,
                )

                pos_ligand_pred = out["coords_pred"].squeeze()
                atoms_pred = out["atoms_pred"]
                if self.hparams.addNumHs and self.hparams.addHybridization:
                    a, b, c, d = (
                        self.num_atom_types,
                        self.num_charge_classes,
                        self.num_Hs,
                        self.num_hybridization,
                    )
                    atoms_pred, charges_pred, numHs_pred, hybridization_pred = (
                        atoms_pred.split([a, b, c, d], dim=-1)
                    )
                    numHs_pred = numHs_pred.softmax(dim=-1)
                    hybridization_pred = hybridization_pred.softmax(dim=-1)
                elif self.hparams.addNumHs and not self.hparams.addHybridization:
                    a, b, c = self.num_atom_types, self.num_charge_classes, self.num_Hs
                    atoms_pred, charges_pred, numHs_pred = atoms_pred.split(
                        [a, b, c], dim=-1
                    )
                    hybridization_pred = None
                    numHs_pred = numHs_pred.softmax(dim=-1)
                elif self.hparams.addHybridization and not self.hparams.addNumHs:
                    a, b, c = (
                        self.num_atom_types,
                        self.num_charge_classes,
                        self.num_hybridization,
                    )
                    atoms_pred, charges_pred, hybridization_pred = atoms_pred.split(
                        [a, b, c], dim=-1
                    )
                    numHs_pred = None
                    hybridization_pred = hybridization_pred.softmax(dim=-1)
                else:
                    a, b = self.num_atom_types, self.num_charge_classes
                    atoms_pred, charges_pred = atoms_pred.split([a, b], dim=-1)
                    numHs_pred = hybridization_pred = None

                atoms_ligand_pred = atoms_pred.softmax(dim=-1)
                # N x a_0
                edges_ligand_pred = out["bonds_pred"].softmax(dim=-1)
                # E x b_0
                charges_ligand_pred = charges_pred.softmax(dim=-1)

                # reverse step

                if pos_noise:
                    wt = noise_fnc(t_node)
                    noise_scale = wt.sqrt()
                else:
                    wt = 1.0
                    noise_scale = 0.0

                if step < len(chain) - 1:
                    pos_ligand = self.position_flow.reverse_sample(
                        xt=pos_ligand,
                        x1_pred=pos_ligand_pred,
                        dt=dt,
                        t=t_node,
                        noise=int(pos_noise),
                        scale=noise_scale,
                        dt_pow=dt_pow,
                    )
                    if score_dynamics:
                        wt = 1.0
                        pos_ligand = (
                            pos_ligand + wt * score_scale * out["score_pred"] * dt
                        )

                atom_types_ligand = atom_types_ligand.argmax(dim=-1, keepdim=True)
                charge_types_ligand = charge_types_ligand.argmax(dim=-1, keepdim=True)
                edge_attr_ligand = edge_attr_ligand.argmax(dim=-1, keepdim=True)

                if i < len(chain) - 1 and cat_noise and self.cat_prior == "uniform":
                    noise = 1
                else:
                    noise = 0

                if discrete_gat:
                    at0 = self.cat_atoms.sample_prior(
                        size=(n,),
                        empirical=self.cat_prior == "empirical",
                        device=device,
                    )
                    ch0 = self.cat_charge.sample_prior(
                        size=(n,),
                        empirical=self.cat_prior == "empirical",
                        device=device,
                    )
                else:
                    at0 = ch0 = None

                atom_types_ligand = self.cat_atoms.reverse_sample(
                    xt=atom_types_ligand,
                    x1_pred=atoms_ligand_pred,
                    x0=at0,
                    t=t_node,
                    noise=noise,
                    dt=dt,
                    mode="gat" if discrete_gat else "cmtc",
                )

                charge_types_ligand = self.cat_charge.reverse_sample(
                    xt=charge_types_ligand,
                    x1_pred=charges_ligand_pred,
                    x0=ch0,
                    t=t_node,
                    noise=noise,
                    dt=dt,
                    mode="gat" if discrete_gat else "cmtc",
                )

                # edges prior
                if discrete_gat:
                    _, edges_prior = wrap_sample_edge_prior(
                        self.cat_bonds, self.cat_prior, edge_index_ligand, n, device
                    )
                else:
                    edges_prior = None

                edge_attr_ligand = self.cat_bonds.reverse_sample(
                    xt=edge_attr_ligand,
                    x1_pred=edges_ligand_pred,
                    x0=edges_prior,
                    t=t_edge,
                    noise=noise,
                    dt=dt,
                    mode="gat" if discrete_gat else "cmtc",
                )
                edge_index_ligand, edge_attr_ligand = symmetrize_edge_attr(
                    edge_index_ligand, edge_attr_ligand
                )

                # deterministic clash guidance
                if clash_guidance:
                    if clash_guidance_start <= i <= clash_guidance_end:
                        _, delta = gd.pocket_clash_guidance(
                            pos_ligand=pos_ligand,
                            pos_pocket=pos_pocket,
                            batch_ligand=batch_ligand,
                            batch_pocket=batch_pocket,
                            sigma=2.0,
                        )
                        pos_ligand = pos_ligand + clash_guidance_scale * delta
                        with torch.enable_grad():
                            pos_ligand = pos_ligand.detach().requires_grad_(True)
                            energy = ligand_pocket_clash_energy_loss(
                                pos_ligand=pos_ligand,
                                pos_pocket=pos_pocket,
                                x_ligand=None,
                                x_pocket=None,
                                min_distance=min_distance,
                                batch_ligand=batch_ligand,
                                batch_pocket=batch_pocket,
                            )
                            delta = torch.autograd.grad(
                                energy.sum(), pos_ligand, create_graph=False
                            )[0]
                            pos_ligand = pos_ligand - clash_guidance_scale * delta
                            pos_ligand = pos_ligand.detach()

                ligand_feats: Dict[str, Tensor] = {
                    "atom_types": atom_types_ligand,
                    "charges": charge_types_ligand,
                    "pos": pos_ligand,
                    "batch": batch_ligand,
                }

                if self.hparams.addNumHs:
                    numHs_ligand = numHs_ligand.argmax(dim=-1, keepdim=True)
                    numHs_ligand = self.cat_numHs.reverse_sample(
                        xt=numHs_ligand,
                        x1_pred=numHs_pred,
                        t=t_node,
                        noise=noise,
                        dt=dt,
                        mode="gat" if discrete_gat else "cmtc",
                    )
                    ligand_feats["numHs"] = numHs_ligand

                if self.hparams.addHybridization:
                    if discrete_gat:
                        h0 = self.cat_hybridization.sample_prior(
                            size=(n,),
                            device=device,
                            empirical=self.cat_prior == "empirical",
                        )
                    else:
                        h0 = None

                    hybridization_ligand = hybridization_ligand.argmax(
                        dim=-1, keepdim=True
                    )
                    hybridization_ligand = self.cat_hybridization.reverse_sample(
                        xt=hybridization_ligand,
                        x1_pred=hybridization_pred,
                        x0=h0,
                        t=t_node,
                        noise=noise,
                        dt=dt,
                        mode="gat" if discrete_gat else "cmtc",
                    )
                    ligand_feats["hybridization"] = hybridization_ligand

                # combine and get PL-complex features on node- and edge-level
                (
                    pl_feats_dict,
                    edge_index_global,
                    edge_attr_global,
                    batch_edge_global,
                    edge_mask_ligand,
                    edge_initial_interaction,
                ) = ut.combine_protein_ligand_feats(
                    ligand_feats,
                    pocket_feats,
                    edge_attr_ligand,
                    self.num_bond_classes,
                    self.cutoff_p,
                    self.cutoff_lp,
                )

                if save_traj:
                    coords_traj.append(
                        pos_ligand.detach().cpu().numpy()
                        + com[batch_ligand].cpu().numpy()
                    )
                    atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
                    edges_traj.append(edge_attr_ligand.detach().cpu().numpy())
                    edges_pred_traj.append(edges_ligand_pred.cpu().numpy())

        # Move generated molecule back to the original pocket position for docking
        pos_ligand += com[batch_ligand]
        out_dict = {
            "coords_pred": pos_ligand,
            "atoms_pred": (atom_types_ligand),
            "charges_pred": (charge_types_ligand),
            "bonds_pred": (edge_attr_ligand),
        }

        if self.hparams.addNumHs:
            out_dict.update({"numHs_pred": numHs_ligand})

        if self.hparams.addHybridization:
            out_dict.update(
                {
                    "hybridization_pred": hybridization_ligand,
                }
            )

        # clashes
        clash_loss = ligand_pocket_clash_energy_loss(
            pos_ligand=pos_ligand.to(device),
            pos_pocket=pocket_data.pos_pocket.to(device),
            batch_ligand=batch_ligand,
            batch_pocket=pocket_data.pos_pocket_batch.to(device),
            x_ligand=(
                atom_types_ligand.argmax(dim=-1)
                if atom_types_ligand.ndim == 2
                else atom_types_ligand
            ),
            x_pocket=pocket_data.x_pocket.to(device),
            count=True,
        ).float()
        clash_loss = clash_loss.mean(0).detach().item()

        molecules = get_generated_molecules(
            out=out_dict,
            data_batch=batch_ligand,
            edge_index_global_lig=edge_index_ligand,
            dataset_info=self.dataset_info,
            device=device,
            mol_device="cpu",
            relax_mol=False,
            max_relax_iter=None,
            sanitize=sanitize,
            build_obabel_mol=False,
        )

        return (
            molecules,
            {
                "coords": coords_traj,
                "atoms": atoms_traj,
                "edges": edges_traj,
                "edges_pred": edges_pred_traj,
            },
            clash_loss,
        )

    def reverse_sampling_node_level_t(
        self,
        pocket_data: Tensor,
        device: str,
        N: Optional[int] = None,
        verbose: bool = False,
        z: Optional[Tensor] = None,
        latent_gamma: float = 1.0,
        save_traj: bool = False,
        clash_guidance: bool = False,
        clash_guidance_start=None,
        clash_guidance_end=None,
        clash_guidance_scale: float = 0.1,
        cat_noise: bool = True,
        pos_noise: bool = False,
        sanitize=False,
        score_dynamics: bool = False,
        score_scale: float = 0.5,
        dt_pow: float = 0.5,
        pos_prior: Optional[Tensor] = None,
        pos_context_noise: float = 0.0,
        noise_schedule: str | Callable = "constant-one",
        discrete_gat: bool = False,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:
        """Implements conditional ligand generation with inpainting"""

        if isinstance(noise_schedule, str):
            assert noise_schedule in ["sine", "brownian", "constant-one"]
            noise_fnc = get_diffusion_callable(noise_schedule=noise_schedule)
        else:
            assert isinstance(noise_schedule, Callable)
            noise_fnc = noise_schedule

        if score_dynamics:
            assert self.hparams.model_score

        pos_pocket = pocket_data.pos_pocket.to(device)
        batch_pocket = pocket_data.pos_pocket_batch.to(device)
        x_pocket = pocket_data.x_pocket.to(device)

        batch_ligand = pocket_data.batch

        coords_traj = []
        atoms_traj = []
        edges_traj = []

        # fully-connected edge-index for ligand
        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)
        n = len(batch_ligand)

        atom_types_ligand = self.cat_atoms.sample_prior(
            size=(n,), empirical=self.cat_prior == "empirical", device=device
        )
        charge_types_ligand = self.cat_charge.sample_prior(
            size=(n,), empirical=self.cat_prior == "empirical", device=device
        )
        edge_attr_ligand = self.cat_bonds.sample_prior(
            size=(n, n), empirical=self.cat_prior == "empirical", device=device
        )
        edge_attr_ligand = edge_attr_ligand[
            edge_index_ligand[0], edge_index_ligand[1], :
        ]
        edge_index_ligand, edge_attr_ligand = symmetrize_edge_attr(
            edge_index_ligand, edge_attr_ligand
        )

        pos_ligand_initial: Tensor = pocket_data.pos.to(device)
        batch_ligand_initial: Tensor = pocket_data.batch.to(device)
        if not self.hparams.com_ligand:
            pos_ligand, pos_pocket, com = remove_mean_pocket(
                pos_ligand_initial, pos_pocket, batch_ligand_initial, batch_pocket
            )
        else:
            raise ValueError("Center of mass ligand shouldnt be used")

        if hasattr(pocket_data, "pos_ligand"):
            pos_ligand_initial = pocket_data.pos_ligand.to(device)
        elif hasattr(pocket_data, "pos"):
            pos_ligand_initial = pocket_data.pos.to(device)
        else:
            raise ValueError("No ligand position provided for inpainting")
        # pos_ligand_initial is assumed to
        # lie within the center of mass of the pocket
        pos_ligand_initial = pos_ligand_initial - com[batch_ligand]
        pos_ligand_initial = pocket_data.data_mask.unsqueeze(-1) * pos_ligand_initial

        # initial features from the fixed ligand
        lig_inpaint_mask = pocket_data.lig_inpaint_mask.to(device)
        lig_inpaint_mask_f = lig_inpaint_mask.float().unsqueeze(-1)
        atom_types_ligand_initial = pocket_data.x.to(device)
        atom_types_ligand_initial = F.one_hot(
            atom_types_ligand_initial, self.num_atom_types
        ).float()
        charges_typed_ligand_initial = pocket_data.charges.to(device)
        charges_typed_ligand_initial = F.one_hot(
            charges_typed_ligand_initial + self.charge_offset, self.num_charge_classes
        ).float()

        if hasattr(pocket_data, "anchor_mask"):
            fragment_mask = (1.0 - lig_inpaint_mask.float()).long()
            anchor_mask = pocket_data.anchor_mask.to(device).long()
            if self.hparams.fragment_prior == "anchor":
                ref_mask = anchor_mask.float()
            elif self.hparams.fragment_prior == "fragment":
                ref_mask = fragment_mask.float()
            ref_points = get_masked_com(
                pos_ligand_initial,
                ref_mask.unsqueeze(-1).float(),
                batch=batch_ligand,
                n_variable=(
                    pocket_data.n_variable
                    if self.hparams.fragment_prior == "fragment"
                    else None
                ),
            )

            anchor_mask_pocket = torch.zeros(
                (len(pos_pocket),), device=device, dtype=torch.long
            )
            fragment_mask_pocket = anchor_mask_pocket.clone()
            anchor_mask = torch.concat((anchor_mask, anchor_mask_pocket), dim=0)
            fragment_mask = torch.concat((fragment_mask, fragment_mask_pocket), dim=0)
            anchor_fragment_embedding = self.anchor_embedding(
                anchor_mask.long()
            ) + self.fragment_embedding(fragment_mask.long())
            fragment_mask = fragment_mask.float().unsqueeze(-1)
        else:
            fragment_mask = anchor_fragment_embedding = None

        if pos_context_noise > 0.0:
            context_noise = torch.randn_like(
                pos_ligand_initial
            ) * pocket_data.data_mask.unsqueeze(-1)
            context_noise = lig_inpaint_mask_f * context_noise
            context_noise_scaled = context_noise * pos_context_noise
            pos_ligand_initial_start = pos_ligand_initial + context_noise_scaled
        else:
            pos_ligand_initial_start = pos_ligand_initial

        # initialize the latent variables / features
        pos_ligand = self.position_flow.sample_prior(n=n, device=device)
        pos_ligand = pos_ligand + ref_points[batch_ligand] * (1.0 - lig_inpaint_mask_f)

        if pos_prior is not None:
            assert torch.all(
                ref_points.mean(0).cpu()
                == torch.zeros(
                    3,
                )
            )
            pos_ligand = pos_ligand + pos_prior[batch_ligand] * (
                1.0 - lig_inpaint_mask_f
            )

        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charge_types_pocket = torch.zeros(
            pos_pocket.shape[0],
            charge_types_ligand.shape[1],
            dtype=torch.float32,
            device=device,
        )

        bond_edge_index = pocket_data.edge_index.to(device)
        bond_edge_attr = pocket_data.edge_attr.to(device)
        edge_index_ligand_true, edge_attr_ligand_true = coalesce_edges(
            edge_index=edge_index_ligand,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_ligand_initial, edge_attr_ligand_initial = sort_edge_index(
            edge_index=edge_index_ligand_true,
            edge_attr=edge_attr_ligand_true,
            sort_by_row=False,
        )
        _, edge_mask_inpainting = get_edge_mask_inpainting(
            edge_index=edge_index_ligand_initial,
            edge_attr=edge_attr_ligand_initial,
            fixed_nodes_indices=pocket_data.lig_inpaint_mask.nonzero().squeeze(),
        )
        edge_mask_f = edge_mask_inpainting.float().unsqueeze(-1)
        edge_attr_ligand_initial = F.one_hot(
            edge_attr_ligand_initial.squeeze().long(), self.num_bond_classes
        ).float()

        edge_index_ligand, edge_attr_ligand = sort_edge_index(
            edge_index_ligand, edge_attr_ligand, sort_by_row=False
        )

        # infilling
        # combine
        # start with the initial features
        pos_ligand = (
            pos_ligand * (1.0 - lig_inpaint_mask_f)
            + pos_ligand_initial_start * lig_inpaint_mask_f
        )
        atom_types_ligand = (
            atom_types_ligand * (1.0 - lig_inpaint_mask_f)
            + atom_types_ligand_initial * lig_inpaint_mask_f
        )

        charge_types_ligand = (
            charge_types_ligand * (1.0 - lig_inpaint_mask_f)
            + charges_typed_ligand_initial * lig_inpaint_mask_f
        )

        edge_attr_ligand = (
            edge_attr_ligand * (1.0 - edge_mask_f)
            + edge_attr_ligand_initial * edge_mask_f
        )

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand,
            "charges": charge_types_ligand,
            "pos": pos_ligand,
            "batch": batch_ligand,
        }

        pocket_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_pocket,
            "charges": charge_types_pocket,
            "pos": pos_pocket,
            "batch": batch_pocket,
        }

        if self.hparams.addNumHs:
            numHs_ligand = self.cat_numHs.sample_prior(
                size=(n,), empirical=self.cat_prior == "empirical", device=device
            )
            numHs_ligand_initial = pocket_data.numHs.to(device)
            numHs_ligand_initial = F.one_hot(numHs_ligand_initial, self.num_Hs).float()
            numHs_pocket = torch.zeros(
                len(pos_pocket), self.num_Hs, device=device, dtype=torch.float32
            )
            # in-filling
            numHs_ligand = (
                numHs_ligand * (1.0 - lig_inpaint_mask_f)
                + numHs_ligand_initial * lig_inpaint_mask_f
            )
            ligand_feats["numHs"] = numHs_ligand
            pocket_feats["numHs"] = numHs_pocket

        if self.hparams.addHybridization:
            hybridization_ligand = self.cat_hybridization.sample_prior(
                size=(n,), empirical=self.cat_prior == "empirical", device=device
            )
            hybridization_ligand_initial = pocket_data.hybridization.to(device)
            hybridization_ligand_initial = F.one_hot(
                hybridization_ligand_initial, self.num_hybridization
            ).float()
            hybridization_pocket = torch.zeros(
                len(pos_pocket),
                self.num_hybridization,
                device=device,
                dtype=torch.float32,
            )
            # in-filling
            hybridization_ligand = (
                hybridization_ligand * (1.0 - lig_inpaint_mask_f)
                + hybridization_ligand_initial * lig_inpaint_mask_f
            )
            ligand_feats["hybridization"] = hybridization_ligand
            pocket_feats["hybridization"] = hybridization_pocket

        # combine and get PL-complex features on node- and edge-level
        (
            pl_feats_dict,
            edge_index_global,
            edge_attr_global,
            batch_edge_global,
            edge_mask_ligand,
            edge_initial_interaction,
        ) = ut.combine_protein_ligand_feats(
            ligand_feats,
            pocket_feats,
            edge_attr_ligand,
            self.num_bond_classes,
            self.cutoff_p,
            self.cutoff_lp,
        )

        eps = 1e-4
        chain = torch.linspace(0 + eps, 1 - eps, N, device=device)
        iterator = tqdm(range(N), total=N) if verbose else range(N)
        t_inpaint_one = torch.ones_like(lig_inpaint_mask_f).float() - eps  # (N, 1)
        # Clash guidance also to be tested #
        if clash_guidance_start is None:
            clash_guidance_start = 0
        if clash_guidance_end is None:
            clash_guidance_end = N

        if save_traj:
            coords_traj.append(
                pos_ligand.detach().cpu().numpy()
                + com[batch_ligand].cpu().numpy()
                # + ref_points[batch_ligand].cpu().numpy()
            )
            atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
            edges_traj.append(edge_attr_ligand.detach().cpu().numpy())

        dt = 1 / len(chain)
        for step, i in enumerate(iterator):

            t = chain[i]  # scalar
            t_node = torch.zeros((n, 1), device=device).fill_(t.item())  # node-level

            t_node = t_inpaint_one * lig_inpaint_mask_f.float() + t_node * (
                1 - lig_inpaint_mask_f.float()
            )
            t_pos_node = t_node.clone()
            variable_mask = fragment_mask
            if pos_context_noise > 0.0:
                t_pos_node = torch.zeros((n, 1), device=device).fill_(t.item())
                variable_mask = pl_feats_dict["ligand_mask"].float().unsqueeze(-1)

            assert t_node.size(0) == batch_ligand.size(
                0
            ), "timesteps must be equal to the number of nodes for ligands"
            t_pocket = scatter_mean(t_pos_node, batch_ligand, dim=0)
            t_pocket = t_pocket[batch_pocket]
            t_ = torch.concat([t_pos_node, t_pocket], dim=0)

            assert t_node.size(0) == batch_ligand.size(
                0
            ), "t and batch must have the same size"
            tmp_t = t_node.squeeze()
            _j, _i = edge_index_ligand
            t_edge = (tmp_t.view(1, -1) + tmp_t.view(-1, 1)) / 2.0
            t_edge = t_edge[_j, _i].unsqueeze(-1)
            # need to populate the visible fixed time variable here as well.
            # the time variable for the nodes also need to be included

            # time embedding
            temb = t_
            node_feats_in = torch.cat(
                [pl_feats_dict[feat] for feat in self.select_node_order], dim=-1
            )

            # field representations
            if self.hparams.include_field_repr:
                v0 = pocket_clash_guidance(
                    pos_ligand=pos_ligand,
                    pos_pocket=pos_pocket,
                    batch_ligand=batch_ligand,
                    batch_pocket=batch_pocket,
                )[1].detach()
                v0 = clamp_norm(v0, maxnorm=10.0)
                v1 = pseudo_lj_guidance(
                    pos_ligand=pos_ligand,
                    pos_pocket=pos_pocket,
                    batch_ligand=batch_ligand,
                    batch_pocket=batch_pocket,
                )[1].detach()
                v1 = clamp_norm(v1, maxnorm=10.0)
                v_01 = torch.stack([v0, v1], dim=-1)
                v_ligand = torch.zeros(
                    (v_01.size(0), 3, self.hparams.sdim), device=device
                )
                v_ligand[:, :, :2] = v_01
                v_pocket = torch.zeros(
                    (pos_pocket.size(0), 3, self.hparams.sdim), device=device
                )
                v = torch.cat([v_ligand, v_pocket], dim=0)
            else:
                v = None

            out = self.model(
                x=node_feats_in,
                t=temb,
                v=v,
                pos=pl_feats_dict["pos"],
                edge_index=edge_index_global,
                edge_attr=edge_attr_global,
                edge_index_ligand=edge_index_ligand,
                batch=pl_feats_dict["batch"],
                edge_attr_initial_ohe=edge_initial_interaction,
                batch_edge_global=batch_edge_global,
                z=z,
                ligand_mask=pl_feats_dict["ligand_mask"],
                batch_ligand=batch_ligand,
                latent_gamma=latent_gamma,
                edge_mask_ligand=edge_mask_ligand,
                anchor_fragment_embedding=anchor_fragment_embedding,
                variable_mask=variable_mask,
            )

            pos_ligand_pred = out["coords_pred"].squeeze()
            atoms_pred = out["atoms_pred"]
            if self.hparams.addNumHs and self.hparams.addHybridization:
                a, b, c, d = (
                    self.num_atom_types,
                    self.num_charge_classes,
                    self.num_Hs,
                    self.num_hybridization,
                )
                atoms_pred, charges_pred, numHs_pred, hybridization_pred = (
                    atoms_pred.split([a, b, c, d], dim=-1)
                )
                numHs_pred = numHs_pred.softmax(dim=-1)
                hybridization_pred = hybridization_pred.softmax(dim=-1)
            elif self.hparams.addNumHs and not self.hp.addHybridization:
                a, b, c = self.num_atom_types, self.num_charge_classes, self.num_Hs
                atoms_pred, charges_pred, numHs_pred = atoms_pred.split(
                    [a, b, c], dim=-1
                )
                hybridization_pred = None
                numHs_pred = numHs_pred.softmax(dim=-1)
            elif self.hparams.addHybridization and not self.hparams.addNumHs:
                a, b, c = (
                    self.num_atom_types,
                    self.num_charge_classes,
                    self.num_hybridization,
                )
                atoms_pred, charges_pred, hybridization_pred = atoms_pred.split(
                    [a, b, c], dim=-1
                )
                numHs_pred = None
                hybridization_pred = hybridization_pred.softmax(dim=-1)
            else:
                a, b = self.num_atom_types, self.num_charge_classes
                atoms_pred, charges_pred = atoms_pred.split([a, b], dim=-1)
                numHs_pred = hybridization_pred = None

            atoms_ligand_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_ligand_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_ligand_pred = charges_pred.softmax(dim=-1)

            # reverse step

            if pos_noise:
                wt = noise_fnc(t_node)
                noise_scale = wt.sqrt()
            else:
                wt = 1.0
                noise_scale = 0.0

            if step < len(chain) - 1:
                pos_ligand = self.position_flow.reverse_sample(
                    xt=pos_ligand,
                    x1_pred=pos_ligand_pred,
                    dt=dt,
                    t=t_pos_node,
                    noise=int(pos_noise),
                    scale=noise_scale,
                    dt_pow=dt_pow,
                )
                if score_dynamics:
                    wt = 1.0
                    pos_ligand = pos_ligand + wt * score_scale * out["score_pred"] * dt

            atom_types_ligand = atom_types_ligand.argmax(dim=-1, keepdim=True)
            charge_types_ligand = charge_types_ligand.argmax(dim=-1, keepdim=True)
            edge_attr_ligand = edge_attr_ligand.argmax(dim=-1, keepdim=True)

            if i < len(chain) - 1 and cat_noise and self.cat_prior == "uniform":
                noise = 1
            else:
                noise = 0

            if discrete_gat:
                at0 = self.cat_atoms.sample_prior(
                    size=(n,),
                    empirical=self.cat_prior == "empirical",
                    device=device,
                )
                ch0 = self.cat_charge.sample_prior(
                    size=(n,),
                    empirical=self.cat_prior == "empirical",
                    device=device,
                )
            else:
                at0 = ch0 = None

            atom_types_ligand = self.cat_atoms.reverse_sample(
                xt=atom_types_ligand,
                x1_pred=atoms_ligand_pred,
                x0=at0,
                t=t_node,
                noise=noise,
                dt=dt,
                mode="gat" if discrete_gat else "cmtc",
            )

            charge_types_ligand = self.cat_charge.reverse_sample(
                xt=charge_types_ligand,
                x1_pred=charges_ligand_pred,
                x0=ch0,
                t=t_node,
                noise=noise,
                dt=dt,
                mode="gat" if discrete_gat else "cmtc",
            )

            # edges prior
            if discrete_gat:
                _, edges_prior = wrap_sample_edge_prior(
                    self.cat_bonds, self.cat_prior, edge_index_ligand, n, device
                )
            else:
                edges_prior = None

            edge_attr_ligand = self.cat_bonds.reverse_sample(
                xt=edge_attr_ligand,
                x1_pred=edges_ligand_pred,
                x0=edges_prior,
                t=t_edge,
                noise=noise,
                dt=dt,
                mode="gat" if discrete_gat else "cmtc",
            )
            edge_index_ligand, edge_attr_ligand = symmetrize_edge_attr(
                edge_index_ligand, edge_attr_ligand
            )

            # in-filling
            # combine
            if torch.isnan(pos_ligand).any():
                print(f"NaN in pos_ligand, step = {step}, i = {i}")
            pos_ligand[pos_ligand.isnan()] = 0.0

            if pos_context_noise == 0.0:
                pos_ligand = (
                    pos_ligand * (1.0 - lig_inpaint_mask_f)
                    + pos_ligand_initial_start * lig_inpaint_mask_f
                )

            atom_types_ligand = (
                atom_types_ligand * (1.0 - lig_inpaint_mask_f)
                + atom_types_ligand_initial * lig_inpaint_mask_f
            )

            charge_types_ligand = (
                charge_types_ligand * (1.0 - lig_inpaint_mask_f)
                + charges_typed_ligand_initial * lig_inpaint_mask_f
            )

            edge_attr_ligand = (
                edge_attr_ligand * (1.0 - edge_mask_f)
                + edge_attr_ligand_initial * edge_mask_f
            )

            # deterministic clash guidance
            if clash_guidance:
                if (
                    clash_guidance_start <= i <= clash_guidance_end
                    and i <= len(chain) - 1
                ):
                    _, delta = gd.pocket_clash_guidance(
                        pos_ligand=pos_ligand,
                        pos_pocket=pos_pocket,
                        batch_ligand=batch_ligand,
                        batch_pocket=batch_pocket,
                        sigma=2.0,
                    )

                    pos_ligand = pos_ligand + clash_guidance_scale * delta
                    if pos_context_noise == 0.0:
                        pos_ligand = (
                            pos_ligand * (1.0 - lig_inpaint_mask_f)
                            + pos_ligand_initial * lig_inpaint_mask_f
                        )

            ligand_feats: Dict[str, Tensor] = {
                "atom_types": atom_types_ligand,
                "charges": charge_types_ligand,
                "pos": pos_ligand,
                "batch": batch_ligand,
            }

            if self.hparams.addNumHs:
                if discrete_gat:
                    h0 = self.cat_numHs.sample_prior(
                        size=(n,),
                        empirical=self.cat_prior == "empirical",
                        device=device,
                    )
                else:
                    h0 = None

                numHs_ligand = numHs_ligand.argmax(dim=-1, keepdim=True)
                numHs_ligand = self.cat_numHs.reverse_sample(
                    xt=numHs_ligand,
                    x1_pred=numHs_pred,
                    x0=h0,
                    t=t_node,
                    noise=noise,
                    dt=dt,
                    mode="gat" if discrete_gat else "cmtc",
                )
                # in-filling
                numHs_ligand = (
                    numHs_ligand * (1.0 - lig_inpaint_mask_f)
                    + numHs_ligand_initial * lig_inpaint_mask_f
                )
                ligand_feats["numHs"] = numHs_ligand

            if self.hparams.addHybridization:
                hybridization_ligand = hybridization_ligand.argmax(dim=-1, keepdim=True)
                if discrete_gat:
                    h0 = self.cat_hybridization.sample_prior(
                        size=(n,),
                        empirical=self.cat_prior == "empirical",
                        device=device,
                    )
                else:
                    h0 = None

                hybridization_ligand = self.cat_hybridization.reverse_sample(
                    xt=hybridization_ligand,
                    x1_pred=hybridization_pred,
                    x0=h0,
                    t=t_node,
                    noise=noise,
                    dt=dt,
                    mode="gat" if discrete_gat else "cmtc",
                )
                # in-filling
                hybridization_ligand = (
                    hybridization_ligand * (1.0 - lig_inpaint_mask_f)
                    + hybridization_ligand_initial * lig_inpaint_mask_f
                )
                ligand_feats["hybridization"] = hybridization_ligand

            # combine and get PL-complex features on node- and edge-level
            (
                pl_feats_dict,
                edge_index_global,
                edge_attr_global,
                batch_edge_global,
                edge_mask_ligand,
                edge_initial_interaction,
            ) = ut.combine_protein_ligand_feats(
                ligand_feats,
                pocket_feats,
                edge_attr_ligand,
                self.num_bond_classes,
                self.cutoff_p,
                self.cutoff_lp,
            )

            if save_traj:
                coords_traj.append(
                    pos_ligand.detach().cpu().numpy()
                    + com[batch_ligand].cpu().numpy()
                    #    + ref_points[batch_ligand].cpu().numpy()
                )
                atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
                edges_traj.append(edge_attr_ligand.detach().cpu().numpy())

        # Move generated molecule back to the original pocket position for docking
        pos_ligand += com[batch_ligand]
        # if hasattr(pocket_data, "anchor_mask"):
        #    pos_ligand += ref_points[batch_ligand]

        # clashes
        clash_loss = ligand_pocket_clash_energy_loss(
            pos_ligand=pos_ligand.to(device),
            pos_pocket=pocket_data.pos_pocket.to(device),
            batch_ligand=batch_ligand,
            batch_pocket=pocket_data.pos_pocket_batch.to(device),
            x_ligand=(
                atom_types_ligand.argmax(dim=-1)
                if atom_types_ligand.ndim == 2
                else atom_types_ligand
            ),
            x_pocket=pocket_data.x_pocket.to(device),
            count=True,
        ).float()
        clash_loss = clash_loss.mean(0).detach().item()

        out_dict = {
            "coords_pred": pos_ligand,
            "atoms_pred": atom_types_ligand,
            "charges_pred": charge_types_ligand,
            "bonds_pred": edge_attr_ligand,
        }
        if self.hparams.addNumHs:
            out_dict.update({"numHs_pred": numHs_ligand})

        if self.hparams.additional_feats:
            out_dict.update(
                {
                    "hybridization_pred": hybridization_ligand,
                }
            )
        molecules = get_generated_molecules(
            out=out_dict,
            data_batch=batch_ligand,
            edge_index_global_lig=edge_index_ligand,
            dataset_info=self.dataset_info,
            device=device,
            mol_device="cpu",
            relax_mol=False,
            max_relax_iter=None,
            sanitize=sanitize,
            build_obabel_mol=False,
        )

        del pocket_data
        torch.cuda.empty_cache()

        return (
            molecules,
            {
                "coords": coords_traj,
                "atoms": atoms_traj,
                "edges": edges_traj,
            },
            clash_loss,
        )


class EQGATDiffL(nn.Module):
    def __init__(
        self,
        hparams: Dict[str, Any],
        dataset_info: DatasetInfo,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        if "node_level_t" not in hparams.keys():
            hparams["node_level_t"] = False
        if "ot_alignment" not in hparams.keys():
            hparams["ot_alignment"] = False
        if "addNumHs" not in hparams.keys():
            hparams["addNumHs"] = False

        # raw input charges start from -2
        self.charge_offset = 2
        self.hparams = namedtuple("hparams", hparams.keys())(*hparams.values())
        self.dataset_info = dataset_info
        self.num_atom_types = hparams["num_atom_types"]
        self.num_bond_classes = hparams["num_bond_classes"]
        self.num_charge_classes = hparams["num_charge_classes"]
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.select_node_order = ["atom_types", "charges"]
        if self.hparams.addNumHs:
            self.select_node_order.append("numHs")
        if self.hparams.addNumHs:
            self.num_Hs = 5  # [0, 1, 2, 3, 4]
            self.num_atom_features += self.num_Hs

        # additional features
        if hparams["additional_feats"]:
            self.num_is_in_ring = 2  # [False, True]
            self.num_is_aromatic = 2  # [False, True]
            self.num_hybridization = 9  # [..., sp3, sp2, sp, ...]
            self.select_add_order = ["ring", "aromatic", "hybridization"]
            self.select_add_split = [
                self.num_is_in_ring,
                self.num_is_aromatic,
                self.num_hybridization,
            ]
            self.num_atom_features += (
                self.num_is_in_ring + self.num_is_aromatic + self.num_hybridization
            )

        if ckpt_path is None:
            self.model = DenoisingEdgeNetwork(
                num_atom_features=self.num_atom_features,
                num_bond_types=self.num_bond_classes,
                hn_dim=(hparams["sdim"], hparams["vdim"]),
                edge_dim=hparams["edim"],
                cutoff_local=hparams["cutoff_local"],
                num_layers=hparams["num_layers"],
                latent_dim=hparams["latent_dim"],
                atom_mapping=True,
                bond_mapping=True,
                use_out_norm=hparams["use_out_norm"],
                store_intermediate_coords=hparams["store_intermediate_coords"],
                joint_property_prediction=hparams["joint_property_prediction"],
                regression_property=hparams["regression_property"],
                node_level_t=hparams["node_level_t"],
            )
        else:
            print(f"Loading model from ckpt path {ckpt_path}")
            self.model = load_model_from_ckpt(ckpt_path)

        self.T = hparams["timesteps"]
        self.node_level_t = (
            hparams["node_level_t"] if "node_level_t" in hparams.keys() else False
        )

        self.cutoff_p = hparams["cutoff_local"]
        self.cutoff_lp = hparams["cutoff_local"]

    def forward(
        self,
        batch: Batch,
        t: Tensor,
        z: Optional[Tensor] = None,
        latent_gamma: float = 1.0,
        pocket_noise_std: float = 0.1,
    ):

        assert t.ndim == 2, "t must be of shape (N, 1)"
        atom_types_ligand: Tensor = batch.x
        pos_ligand: Tensor = batch.pos
        charges_ligand: Tensor = batch.charges
        batch_ligand: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        N = len(batch_ligand)

        device = pos_ligand.device
        pos_ligand_com = scatter_mean(pos_ligand, batch_ligand, dim=0)
        pos_centered_ligand = pos_ligand - pos_ligand_com[batch_ligand]

        # ligand edges
        # fully-connected ligand
        edge_index_global_lig = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
        edge_index_global_lig = sort_edge_index(
            edge_index_global_lig, sort_by_row=False
        )
        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        # feature for interaction,
        # ligand-ligand
        edge_initial_interaction = torch.zeros(
            (edge_index_global_lig.size(1), 3),
            dtype=torch.float32,
            device=device,
        )
        edge_initial_interaction[:, 0] = 1.0  # ligand-ligand
        batch_edge_global = batch_ligand[edge_index_global_lig[0]]  #

        # one-hot-encode discrete features
        atom_types_ligand = F.one_hot(atom_types_ligand, self.num_atom_types).float()
        charges_ligand = F.one_hot(
            charges_ligand + self.charge_offset, self.num_charge_classes
        ).float()
        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig, self.num_bond_classes
        ).float()

        # sample latent prior
        pos_prior = torch.randn_like(pos_ligand)
        pos_prior = (
            pos_prior - scatter_mean(pos_prior, batch_ligand, dim=0)[batch_ligand]
        )
        # OT
        pos_prior = self.optimal_transport_alignment(
            pos_ligand, pos_prior, batch_ligand
        )

        # assume continuous variables
        atom_types_prior = torch.randn_like(atom_types_ligand)
        charges_prior = torch.randn_like(charges_ligand)
        edges_prior = torch.randn((N, N, self.num_bond_classes), device=device)
        edges_prior = 0.5 * (edges_prior + edges_prior.permute(1, 0, 2))
        edges_prior = edges_prior[edge_index_global_lig[0], edge_index_global_lig[1], :]

        if self.node_level_t:
            _j, _i = edge_index_global_lig
            t_node = t
            assert t.size(0) == batch_ligand.size(
                0
            ), "t and data_batch must have the same size"
            t_ = t.squeeze(dim=1)
            assert t_.ndim == 1, "t must be 1D"
            t_ = (t_.view(1, -1) + t_.view(-1, 1)) / 2.0  # (N, N)
            t_edge = t_[_j, _i].unsqueeze(-1)
        else:
            t_node = t[batch_ligand]
            t_edge = t[batch_edge_global]

        atom_types_ligand_perturbed = (
            atom_types_ligand * t_node + (1.0 - t_node) * atom_types_prior
        )
        charges_ligand_perturbed = (
            charges_ligand * t_node + (1.0 - t_node) * charges_prior
        )
        edge_attr_global_perturbed_lig = (
            edge_attr_global_lig * t_edge + (1.0 - t_edge) * edges_prior
        )
        pos_perturbed_ligand = pos_centered_ligand * t_node + (1.0 - t_node) * pos_prior

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand_perturbed,
            "charges": charges_ligand_perturbed,
            "pos": pos_perturbed_ligand,
            "batch": batch_ligand,
        }

        # Concatenate all node features along the feature dims
        atom_feats_in_perturbed = torch.cat(
            [ligand_feats[feat] for feat in self.select_node_order],
            dim=-1,
        )
        pos_joint_perturbed = ligand_feats["pos"]
        batch_full = ligand_feats["batch"]

        if self.hparams.additional_feats:
            add_feats_perturbed = torch.cat(
                [ligand_feats[feat] for feat in self.select_add_order], dim=-1
            )
            atom_feats_in_perturbed = torch.cat(
                [atom_feats_in_perturbed, add_feats_perturbed], dim=-1
            )

        # forward pass for the model
        out = self.model(
            x=atom_feats_in_perturbed,
            t=t,
            pos=pos_joint_perturbed,
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_perturbed_lig,
            batch=batch_full,
            edge_attr_initial_ohe=edge_initial_interaction,
            batch_edge_global=batch_edge_global,
            z=z,
            batch_ligand=batch_ligand,
            latent_gamma=latent_gamma,
        )
        out["t"] = t

        # Ground truth masking
        out["coords_true"] = pos_centered_ligand
        out["atoms_true"] = atom_types_ligand.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_lig.argmax(dim=-1)
        out["charges_true"] = charges_ligand.argmax(dim=-1)
        out["bond_aggregation_index"] = edge_index_global_lig[1]

        out["numHs_true"] = None
        out["ring_true"] = None
        out["aromatic_true"] = None
        out["hybridization_true"] = None

        return out

    def select_splitted_node_feats(
        self, x: Tensor, batch_num_nodes: Tensor, select: Tensor
    ):
        x_split = x.split(batch_num_nodes.cpu().numpy().tolist(), dim=0)
        x_select = torch.concat([x_split[i] for i in select.cpu().numpy()], dim=0)
        return x_select.to(x.device)

    def _optimal_transport_alignment(self, a: Tensor, b: Tensor):
        C = torch.cdist(a, b, p=2)
        _, dest_ind = linear_sum_assignment(C.cpu().numpy(), maximize=False)
        dest_ind = torch.tensor(dest_ind, device=a.device)
        b_sorted = b[dest_ind]
        return b_sorted

    def optimal_transport_alignment(  # noqa: F811
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

    def reverse_sampling(
        self,
        N: int,
        num_graphs: int,
        device: str,
        num_nodes_lig: Tensor,
        verbose: bool = False,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        latent_gamma: float = 1.0,
        z: Optional[Tensor] = None,
        save_traj: bool = False,
        eps: float = 1e-4,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:

        batch_ligand = torch.arange(num_graphs, device=device).repeat_interleave(
            num_nodes_lig, dim=0
        )
        bs = num_graphs
        coords_traj = []
        atoms_traj = []
        edges_traj = []
        n = len(batch_ligand)

        # initialize the latent variables / features from Gaussian prior
        pos_ligand_prior = torch.randn(n, 3, device=device)
        pos_ligand_prior = (
            pos_ligand_prior
            - scatter_mean(pos_ligand_prior, batch_ligand, dim=0)[batch_ligand]
        )
        atom_types_ligand_prior = torch.randn(n, self.num_atom_types, device=device)
        charge_types_ligand_prior = torch.randn(
            n, self.num_charge_classes, device=device
        )
        edge_attr_ligand_prior = torch.randn(
            (n, n, self.num_bond_classes), device=device
        )
        edge_attr_ligand_prior = 0.5 * (
            edge_attr_ligand_prior + edge_attr_ligand_prior.permute(1, 0, 2)
        )

        pos_ligand = pos_ligand_prior.clone()
        atom_types_ligand = atom_types_ligand_prior.clone()
        charge_types_ligand = charge_types_ligand_prior.clone()
        edge_attr_ligand = edge_attr_ligand_prior.clone()

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand,
            "charges": charge_types_ligand,
            "pos": pos_ligand,
            "batch": batch_ligand,
        }

        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)
        edge_attr_ligand = edge_attr_ligand[
            edge_index_ligand[0], edge_index_ligand[1], :
        ]
        edge_initial_interaction = torch.zeros(
            (edge_index_ligand.size(1), 3),
            dtype=torch.float32,
            device=device,
        )
        edge_initial_interaction[:, 0] = 1.0  # ligand-ligand
        batch_edge_global = batch_ligand[edge_index_ligand[0]]  #

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand,
            "charges": charge_types_ligand,
            "pos": pos_ligand,
            "batch": batch_ligand,
        }

        chain = torch.linspace(0 + eps, 1 - eps, N, device=device)
        iterator = tqdm(range(N), total=N) if verbose else range(N)
        for i in iterator:
            t = chain[i]
            s = t + (1 / N)
            coeff_x1 = ((s - t) / (1 - t)).clamp(min=0.0, max=1.0).item()
            coeff_xt = ((1 - s) / (1 - t)).clamp(min=0.0, max=1.0).item()
            temb = torch.zeros((bs, 1), device=device).fill_(t.item())
            node_feats_in = torch.cat(
                [ligand_feats[feat] for feat in self.select_node_order], dim=-1
            )
            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=ligand_feats["pos"],
                edge_index=edge_index_ligand,
                edge_attr=edge_attr_ligand,
                batch=ligand_feats["batch"],
                edge_attr_initial_ohe=edge_initial_interaction,
                batch_edge_global=batch_edge_global,
                z=z,
                latent_gamma=latent_gamma,
            )
            pos_ligand_pred = out["coords_pred"].squeeze()
            atoms_pred = out["atoms_pred"]
            atoms_pred, charges_pred = atoms_pred.split(
                [self.num_atom_types, self.num_charge_classes],
                dim=-1,
            )
            atoms_ligand_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_ligand_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_ligand_pred = charges_pred.softmax(dim=-1)

            pos_ligand = coeff_x1 * pos_ligand_pred + coeff_xt * pos_ligand
            atom_types_ligand = (
                coeff_x1 * atoms_ligand_pred + coeff_xt * atom_types_ligand
            )
            charge_types_ligand = (
                coeff_x1 * charges_ligand_pred + coeff_xt * charge_types_ligand
            )
            edge_attr_ligand = (
                coeff_x1 * edges_ligand_pred + coeff_xt * edge_attr_ligand
            )

            if save_traj:
                coords_traj.append(pos_ligand.cpu().numpy())
                atoms_traj.append(atom_types_ligand.argmax(-1).cpu().numpy())
                edges_traj.append(edge_attr_ligand.argmax(-1).cpu().numpy())

            ligand_feats: Dict[str, Tensor] = {
                "atom_types": atom_types_ligand,
                "charges": charge_types_ligand,
                "pos": pos_ligand,
                "batch": batch_ligand,
            }

        out_dict = {
            "coords_pred": pos_ligand,
            "atoms_pred": atom_types_ligand,
            "charges_pred": charge_types_ligand,
            "bonds_pred": edge_attr_ligand,
        }

        molecules = get_generated_molecules(
            out_dict,
            batch_ligand,
            edge_index_ligand,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            device=device,
            mol_device="cpu",
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            while_train=False,
            build_obabel_mol=build_obabel_mol,
        )

        return molecules, {
            "coords": coords_traj,
            "atoms": atoms_traj,
            "edges": edges_traj,
        }

    def reverse_sampling_node_level_t(
        self,
        pocket_data: Batch,
        N: int,
        num_graphs: int,
        device: str,
        num_nodes_lig: Tensor,
        verbose: bool = False,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        latent_gamma: float = 1.0,
        z: Optional[Tensor] = None,
        save_traj: bool = False,
        eps: float = 1e-4,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:

        batch_ligand = pocket_data.batch
        coords_traj = []
        atoms_traj = []
        edges_traj = []

        if hasattr(pocket_data, "pos_ligand"):
            pos_ligand_initial = pocket_data.pos_ligand.to(device)
        elif hasattr(pocket_data, "pos"):
            pos_ligand_initial = pocket_data.pos.to(device)
        else:
            raise ValueError("No ligand position provided for inpainting")

        # initial features from the fixed ligand
        lig_inpaint_mask = pocket_data.lig_inpaint_mask.to(device)
        lig_inpaint_mask_f = lig_inpaint_mask.float().unsqueeze(-1)
        atom_types_ligand_initial = pocket_data.x.to(device)
        atom_types_ligand_initial = F.one_hot(
            atom_types_ligand_initial, self.num_atom_types
        ).float()
        charge_types_ligand_initial = pocket_data.charges.to(device)
        charge_types_ligand_initial = F.one_hot(
            charge_types_ligand_initial + self.charge_offset, self.num_charge_classes
        ).float()
        bond_edge_index = pocket_data.edge_index.to(device)
        bond_edge_attr = pocket_data.edge_attr.to(device)
        n = len(pos_ligand_initial)

        # initialize the latent variables / features from Gaussian prior
        pos_ligand_prior = torch.randn(n, 3, device=device)
        pos_ligand_prior = (
            pos_ligand_prior
            - scatter_mean(pos_ligand_prior, batch_ligand, dim=0)[batch_ligand]
        )
        atom_types_ligand_prior = torch.randn(n, self.num_atom_types, device=device)
        charge_types_ligand_prior = torch.randn(
            n, self.num_charge_classes, device=device
        )
        edge_attr_ligand_prior = torch.randn(
            (n, n, self.num_bond_classes), device=device
        )
        edge_attr_ligand_prior = 0.5 * (
            edge_attr_ligand_prior + edge_attr_ligand_prior.permute(1, 0, 2)
        )

        # fully-connected edge-index for ligand
        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)

        edge_index_ligand_true, edge_attr_ligand_true = coalesce_edges(
            edge_index=edge_index_ligand,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_ligand_initial, edge_attr_ligand_initial = sort_edge_index(
            edge_index=edge_index_ligand_true,
            edge_attr=edge_attr_ligand_true,
            sort_by_row=False,
        )
        _, edge_mask_inpainting = get_edge_mask_inpainting(
            edge_index=edge_index_ligand_initial,
            edge_attr=edge_attr_ligand_initial,
            fixed_nodes_indices=pocket_data.lig_inpaint_mask.nonzero().squeeze(),
        )
        edge_mask_f = edge_mask_inpainting.float().unsqueeze(-1)
        edge_attr_ligand_initial = F.one_hot(
            edge_attr_ligand_initial.squeeze().long(), self.num_bond_classes
        ).float()

        edge_attr_ligand_prior = edge_attr_ligand_prior[
            edge_index_ligand_initial[0], edge_index_ligand_initial[1], :
        ]
        # feature for interaction,
        # ligand-ligand
        edge_initial_interaction = torch.zeros(
            (edge_index_ligand.size(1), 3),
            dtype=torch.float32,
            device=device,
        )
        edge_initial_interaction[:, 0] = 1.0  # ligand-ligand
        batch_edge_global = batch_ligand[edge_index_ligand[0]]  #

        # infilling

        # combine
        pos_ligand = (
            pos_ligand_prior * (1.0 - lig_inpaint_mask_f)
            + pos_ligand_initial * lig_inpaint_mask_f
        )
        atom_types_ligand = (
            atom_types_ligand_prior * (1.0 - lig_inpaint_mask_f)
            + atom_types_ligand_initial * lig_inpaint_mask_f
        )

        charge_types_ligand = (
            charge_types_ligand_prior * (1.0 - lig_inpaint_mask_f)
            + charge_types_ligand_initial * lig_inpaint_mask_f
        )

        edge_attr_ligand = (
            edge_attr_ligand_prior * (1.0 - edge_mask_f)
            + edge_attr_ligand_initial * edge_mask_f
        )

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand,
            "charges": charge_types_ligand,
            "pos": pos_ligand,
            "batch": batch_ligand,
        }

        chain = torch.linspace(0 + eps, 1 - eps, N, device=device)
        iterator = tqdm(range(N), total=N) if verbose else range(N)
        t_inpaint_one = torch.ones_like(lig_inpaint_mask_f).float() - eps  # (N, 1)

        for i in iterator:

            t = chain[i]  # scalar
            t = torch.zeros((n, 1), device=device).fill_(t.item())  # node-level
            t = t_inpaint_one * lig_inpaint_mask_f.float() + t * (
                1 - lig_inpaint_mask_f.float()
            )
            s = t + (1 / N)

            coeff_x1 = ((s - t) / (1 - t)).clamp(min=0.0, max=1.0)
            coeff_xt = ((1 - s) / (1 - t)).clamp(min=0.0, max=1.0)

            assert t.size(0) == batch_ligand.size(
                0
            ), "timesteps must be equal to the number of nodes for ligands"

            # time embedding
            temb = t

            node_feats_in = torch.cat(
                [ligand_feats[feat] for feat in self.select_node_order], dim=-1
            )

            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=ligand_feats["pos"],
                edge_index=edge_index_ligand,
                edge_attr=edge_attr_ligand,
                edge_index_ligand=edge_index_ligand,
                batch=ligand_feats["batch"],
                edge_attr_initial_ohe=edge_initial_interaction,
                batch_edge_global=batch_edge_global,
                z=z,
                latent_gamma=latent_gamma,
            )

            pos_ligand_pred = out["coords_pred"].squeeze()
            atoms_pred = out["atoms_pred"]

            atoms_pred, charges_pred = atoms_pred.split(
                [self.num_atom_types, self.num_charge_classes],
                dim=-1,
            )

            atoms_ligand_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_ligand_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_ligand_pred = charges_pred.softmax(dim=-1)

            pos_ligand = coeff_x1 * pos_ligand_pred + coeff_xt * pos_ligand
            atom_types_ligand = (
                coeff_x1 * atoms_ligand_pred + coeff_xt * atom_types_ligand
            )
            charge_types_ligand = (
                coeff_x1 * charges_ligand_pred + coeff_xt * charge_types_ligand
            )

            edge_attr_ligand = (
                coeff_x1[batch_edge_global] * edges_ligand_pred
                + coeff_xt[batch_edge_global] * edge_attr_ligand
            )

            # infilling
            # combine, shouldnt be necessary because of the coefficients coeff_x1 and coeff_xt
            pos_ligand[pos_ligand.isnan()] = 0.0
            pos_ligand = (
                pos_ligand * (1.0 - lig_inpaint_mask_f)
                + pos_ligand_initial * lig_inpaint_mask_f
            )
            atom_types_ligand = (
                atom_types_ligand * (1.0 - lig_inpaint_mask_f)
                + atom_types_ligand_initial * lig_inpaint_mask_f
            )
            charge_types_ligand = (
                charge_types_ligand * (1.0 - lig_inpaint_mask_f)
                + charge_types_ligand_initial * lig_inpaint_mask_f
            )

            ligand_feats: Dict[str, Tensor] = {
                "atom_types": atom_types_ligand,
                "charges": charge_types_ligand,
                "pos": pos_ligand,
                "batch": batch_ligand,
            }

            edge_attr_ligand = (
                edge_attr_ligand * (1.0 - edge_mask_f)
                + edge_attr_ligand_initial * edge_mask_f
            )

            if save_traj:
                coords_traj.append(pos_ligand.detach().cpu().numpy())
                atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
                edges_traj.append(edge_attr_ligand.detach().cpu().numpy())

        # Move generated molecule back to the original pocket position for docking
        out_dict = {
            "coords_pred": pos_ligand,
            "atoms_pred": atom_types_ligand,
            "charges_pred": charge_types_ligand,
            "bonds_pred": edge_attr_ligand,
        }

        molecules = get_generated_molecules(
            out_dict,
            batch_ligand,
            edge_index_ligand,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            device=device,
            mol_device="cpu",
            relax_mol=None,
            max_relax_iter=100,
            sanitize=None,
            while_train=False,
            build_obabel_mol=None,
        )

        del pocket_data
        torch.cuda.empty_cache()

        return molecules, {
            "coords": coords_traj,
            "atoms": atoms_traj,
            "edges": edges_traj,
        }

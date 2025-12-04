from collections import namedtuple
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_add, scatter_mean
from tqdm import tqdm

import e3mol.experiments.guidance as gd
import e3mol.experiments.utils as ut
from e3mol.experiments.data.datainfo import DatasetInfo
from e3mol.experiments.data.molecule import Molecule
from e3mol.experiments.diffusion.categorical import CategoricalDiffusionKernel
from e3mol.experiments.diffusion.gaussian import DiscreteDDPM
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


def _create_uniform_prior(arr):
    arr_unif = torch.ones_like(arr)
    arr_unif = arr_unif / torch.sum(arr_unif)
    return arr_unif


class EQGATDiffPL(nn.Module):
    def __init__(
        self,
        hparams: Dict[str, Any],
        dataset_info: DatasetInfo,
        ckpt_path: Optional[str] = None,
        uniform_prior: bool = False,
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

        self.old_conditional_learning = True

        self.hparams = namedtuple("hparams", hparams.keys())(*hparams.values())
        self.dataset_info = dataset_info
        self.num_atom_types = hparams["num_atom_types"]
        self.num_bond_classes = hparams["num_bond_classes"]
        self.num_charge_classes = hparams["num_charge_classes"]
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.select_node_order = ["atom_types", "charges"]

        # data dependent prior
        atom_types_prior = dataset_info.atom_types.float()
        charge_types_prior = dataset_info.charges_marginals.float()
        bond_types_prior = dataset_info.edge_types.float()

        if uniform_prior:
            # uniform prior
            print("Using uniform prior for discrete variables")
            atom_types_prior = _create_uniform_prior(atom_types_prior)
            charge_types_prior = _create_uniform_prior(charge_types_prior)
            bond_types_prior = _create_uniform_prior(bond_types_prior)

            dataset_info.atom_types = atom_types_prior
            dataset_info.charges_marginals = charge_types_prior
            dataset_info.edge_types = bond_types_prior

        self.register_buffer("atoms_prior", atom_types_prior.clone())
        self.register_buffer("bonds_prior", bond_types_prior.clone())
        self.register_buffer("charges_prior", charge_types_prior.clone())

        if self.hparams.addNumHs:
            self.num_Hs = 5  # [0, 1, 2, 3, 4]
            self.num_atom_features += self.num_Hs
            self.select_node_order.append("numHs")
            numHs_prior = dataset_info.numHs.float()
            self.register_buffer("numHs_prior", numHs_prior.clone())

        if self.hparams.addHybridization:
            self.num_hybridization = 9  # [..., sp3, sp2, sp, ...]
            self.num_atom_features += self.num_hybridization
            self.select_node_order.append("hybridization")
            hybridization_prior = dataset_info.hybridization.float()
            self.register_buffer("hybridization_prior", hybridization_prior.clone())

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
                include_field_repr=hparams["include_field_repr"],
            )
        else:
            print(f"Loading model from ckpt path {ckpt_path}")
            self.model = load_model_from_ckpt(ckpt_path, old=False)

        self.T = hparams["timesteps"]
        self.node_level_t = (
            hparams["node_level_t"] if "node_level_t" in hparams.keys() else False
        )

        self.sde_pos = DiscreteDDPM(
            scaled_reverse_posterior_sigma=True,
            schedule=hparams["noise_scheduler"],
            nu=2.5,
            enforce_zero_terminal_snr=False,
            T=hparams["timesteps"],
            clamp_alpha_min=0.05,
        )
        self.sde_atom_charge = DiscreteDDPM(
            scaled_reverse_posterior_sigma=True,
            schedule=hparams["noise_scheduler"],
            nu=1.0,
            enforce_zero_terminal_snr=False,
            T=hparams["timesteps"],
            clamp_alpha_min=0.05,
        )
        self.cat_atoms = CategoricalDiffusionKernel(
            terminal_distribution=atom_types_prior,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_prior,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        if self.hparams.addNumHs:
            self.cat_numHs = CategoricalDiffusionKernel(
                terminal_distribution=numHs_prior,
                alphas=self.sde_atom_charge.alphas.clone(),
            )
        self.sde_bonds = DiscreteDDPM(
            scaled_reverse_posterior_sigma=True,
            schedule=hparams["noise_scheduler"],
            nu=1.5,
            enforce_zero_terminal_snr=False,
            T=hparams["timesteps"],
            clamp_alpha_min=0.05,
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_prior,
            alphas=self.sde_bonds.alphas.clone(),
        )

        if self.hparams.addHybridization:
            self.cat_hybridization = CategoricalDiffusionKernel(
                terminal_distribution=hybridization_prior,
                alphas=self.sde_atom_charge.alphas.clone(),
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
        inpainting: bool = False,
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

            # ligand is shifted into by pocket com, now we shift
            # again on the reference (anchor or variable fragment) com
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
            # old
            if self.old_conditional_learning:
                pos_centered_ligand = (
                    pos_centered_ligand
                    - (ref_point * ((~uncond_case.unsqueeze(-1)).float()))[batch_ligand]
                )
                # pocket is shifted to 0-com, now we shift again on the anchor com
                pos_centered_pocket = (
                    pos_centered_pocket
                    - (ref_point * ((~uncond_case.unsqueeze(-1)).float()))[batch_pocket]
                )

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
            # variable
            fragment_mask_ligand = torch.ones(
                (pos_ligand.shape[0],), device=device, dtype=torch.bool
            )
            ref_point = None

        # forward noising for ligand features: coordinates, atom-types, charges
        _, pos_perturbed_ligand = self.sde_pos.sample_pos(
            t,
            pos_centered_ligand,
            batch_ligand,
            remove_mean=False,
            extend_t=not self.node_level_t or inpainting,
            ot_alignment=self.hparams.ot_alignment,  # optimal transport alignment
        )

        if fragment_mask is not None:
            if not self.old_conditional_learning:
                # shift com to reference points of the variable part in the inpainting case
                pos_perturbed_ligand += (
                    ref_point[batch_ligand]
                    * (~uncond_case)[batch_ligand].unsqueeze(-1).float()
                )
                if self.hparams.ot_alignment:
                    pos_perturbed_ligand = self.sde_pos.optimal_transport_alignment(
                        pos_centered_ligand, pos_perturbed_ligand, batch_ligand
                    )

        atom_types_ligand, atom_types_ligand_perturbed = (
            self.cat_atoms.sample_categorical(
                t,
                atom_types_ligand,
                batch_ligand,
                self.dataset_info,
                num_classes=self.num_atom_types,
                type="atoms",
                extend_t=not self.node_level_t or inpainting,
            )
        )
        charges_ligand, charges_ligand_perturbed = self.cat_charges.sample_categorical(
            t,
            charges_ligand,
            batch_ligand,
            self.dataset_info,
            num_classes=self.num_charge_classes,
            type="charges",
            extend_t=not self.node_level_t or inpainting,
        )

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

        # forward noising edges
        edge_attr_global_perturbed_lig = self.cat_bonds.sample_edges_categorical(
            t,
            edge_index_global_lig,
            edge_attr_global_lig,
            batch_ligand,
            return_one_hot=True,
            extend_t=not self.node_level_t or inpainting,
        )

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

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand_perturbed,
            "charges": charges_ligand_perturbed,
            "pos": pos_perturbed_ligand,
            "batch": batch_ligand,
        }

        pocket_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_pocket,
            "charges": charges_pocket,
            "pos": pos_centered_pocket,
            "batch": batch_pocket,
        }

        if self.hparams.addNumHs:
            numHs_ligand: Tensor = batch.numHs
            numHs_ligand, numHs_ligand_perturbed = self.cat_numHs.sample_categorical(
                t,
                numHs_ligand,
                batch_ligand,
                self.dataset_info,
                num_classes=self.num_Hs,
                type="numHs",
                extend_t=not self.node_level_t or inpainting,
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
            hybridization_ligand: Tensor = batch.hybridization
            hybridization_ligand, hybridization_ligand_perturbed = (
                self.cat_hybridization.sample_categorical(
                    t,
                    hybridization_ligand,
                    batch_ligand,
                    self.dataset_info,
                    num_classes=self.num_hybridization,
                    type="hybridization",
                    extend_t=not self.node_level_t or inpainting,
                )
            )
            hybridization_pocket = torch.zeros(
                pos_pocket.shape[0],
                hybridization_ligand_perturbed.shape[1],
                dtype=torch.float32,
                device=device,
            )
            ligand_adds = {
                "hybridization": hybridization_ligand_perturbed,
            }
            pocket_adds = {
                "hybridization": hybridization_pocket,
            }

            ligand_feats.update(ligand_adds)
            pocket_feats.update(pocket_adds)

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

        if fragment_mask is None:
            fragment_mask = ligand_mask.unsqueeze(-1).float()

        if self.hparams.additional_feats:
            add_feats_perturbed = torch.cat(
                [pl_feats_dict[feat] for feat in self.select_add_order], dim=-1
            )
            atom_feats_in_perturbed = torch.cat(
                [atom_feats_in_perturbed, add_feats_perturbed], dim=-1
            )

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

        if self.node_level_t or inpainting:
            assert t.size(0) == batch_ligand.size(
                0
            ), "timesteps must be equal to the number of nodes for ligands"
            tpocket = scatter_mean(t, batch_ligand, dim=0).long()
            tpocket = tpocket[batch_pocket]
            t = torch.concat([t, tpocket], dim=0)

        # time embedding
        temb = t.float() / self.hparams.timesteps  # type: ignore [attr-defined]
        temb = temb.clamp(min=1e-3)
        temb = temb.unsqueeze(dim=1)

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

        # forward pass for the model
        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
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
            variable_mask=fragment_mask,
        )
        out["t"] = t
        out["ligand_mask"] = ligand_mask
        # loss computation only for variable fragment atoms
        out["variable_mask"] = fragment_mask_ligand
        # Ground truth masking
        out["coords_true"] = pos_centered_ligand
        out["atoms_true"] = atom_types_ligand.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_lig
        out["charges_true"] = charges_ligand.argmax(dim=-1)
        out["bond_aggregation_index"] = edge_index_global_lig[1]

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

    def _inpaint_forward_combine(
        self,
        s: Tensor,
        pos_ligand_initial: Tensor,
        batch_ligand: Tensor,
        atom_types_ligand_initial: Tensor,
        pos_ligand: Tensor,
        lig_inpaint_mask_f: Tensor,
        atom_types_ligand: Tensor,
        edge_index_global_lig: Tensor,
        edge_attr_ligand_initial: Tensor,
        edge_mask_f: Tensor,
        edge_attr_ligand: Tensor,
        inpaint_edges: bool = False,
        hybridization_ligand_initial=None,
        extend_t: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # inpainting through forward noising
        _, pos_ligand_inpaint = self.sde_pos.sample_pos(
            s,
            pos_ligand_initial,
            batch_ligand,
            remove_mean=False,
            extend_t=extend_t,
        )
        _, atom_types_inpaint = self.cat_atoms.sample_categorical(
            s,
            atom_types_ligand_initial,
            batch_ligand,
            self.dataset_info,
            num_classes=self.num_atom_types,
            type="atoms",
            extend_t=extend_t,
        )

        # combine
        pos_ligand = (
            pos_ligand * (1.0 - lig_inpaint_mask_f)
            + pos_ligand_inpaint * lig_inpaint_mask_f
        )
        atom_types_ligand = (
            atom_types_ligand * (1.0 - lig_inpaint_mask_f)
            + atom_types_inpaint * lig_inpaint_mask_f
        )

        if inpaint_edges:

            edge_attr_inpaint = self.cat_bonds.sample_edges_categorical(
                s,
                edge_index_global_lig,
                edge_attr_ligand_initial,
                batch_ligand,
                return_one_hot=True,
                extend_t=extend_t,
            )

            edge_attr_ligand = (
                edge_attr_ligand * (1.0 - edge_mask_f) + edge_attr_inpaint * edge_mask_f
            )

        return pos_ligand, atom_types_ligand, edge_attr_ligand

    def reverse_sampling(
        self,
        num_graphs: int,
        pocket_data: Tensor,
        device: str,
        num_nodes_lig: Optional[Tensor],
        verbose: bool = False,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        latent_gamma: float = 1.0,
        clash_guidance: bool = False,
        clash_guidance_start=None,
        clash_guidance_end=None,
        clash_guidance_scale: float = 0.01,
        save_traj: bool = False,
        inpainting: bool = False,
        emd_ot: bool = False,
        z: Optional[Tensor] = None,
        _pos_ligand: Optional[Tensor] = None,
        probability_flow_ode: bool = False,
        inpaint_edges: bool = False,
        resample_steps: int = 1,
        **kwargs,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:

        pos_pocket = pocket_data.pos_pocket.to(device)
        batch_pocket = pocket_data.pos_pocket_batch.to(device)
        x_pocket = pocket_data.x_pocket.to(device)

        if not inpainting:
            batch_ligand = torch.arange(num_graphs, device=device).repeat_interleave(
                num_nodes_lig, dim=0
            )
        else:
            batch_ligand = pocket_data.batch
            num_nodes_lig = batch_ligand.bincount()

        bs = num_graphs
        coords_traj = []
        atoms_traj = []
        edges_traj = []

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

        pos_ligand = torch.randn((len(batch_ligand), 3), device=device)

        n = len(pos_ligand)

        if self.anchor_feature:
            anchor_mask_ligand = torch.zeros(
                (len(pos_ligand),), device=device, dtype=torch.long
            )
            anchor_mask_pocket = torch.zeros(
                (len(pos_pocket),), device=device, dtype=torch.long
            )
            anchor_mask = torch.cat([anchor_mask_ligand, anchor_mask_pocket], dim=0)
            anchor_embedding = self.anchor_embedding(anchor_mask)

            fragment_mask_ligand = torch.ones(
                (len(pos_ligand),), device=device, dtype=torch.long
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

        # initialize the atom- and charge types
        atom_types_ligand = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types_ligand = F.one_hot(atom_types_ligand, self.num_atom_types).float()

        charge_types_ligand = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types_ligand = F.one_hot(
            charge_types_ligand, self.num_charge_classes
        ).float()

        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charge_types_pocket = torch.zeros(
            pos_pocket.shape[0],
            charge_types_ligand.shape[1],
            dtype=torch.float32,
            device=device,
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
            numHs_ligand = torch.multinomial(
                self.numHs_prior, num_samples=n, replacement=True
            )
            numHs_ligand = F.one_hot(numHs_ligand, self.num_Hs).float()
            num_Hs_pos_pocket = torch.zeros(
                pos_pocket.shape[0],
                numHs_ligand.shape[1],
                dtype=torch.float32,
                device=device,
            )
            ligand_feats["numHs"] = numHs_ligand
            pocket_feats["numHs"] = num_Hs_pos_pocket
        else:
            numHs_ligand = None
            num_Hs_pos_pocket = None

        if self.hparams.addHybridization:
            hybridization_ligand = torch.multinomial(
                self.hybridization_prior, num_samples=n, replacement=True
            )
            hybridization_ligand = F.one_hot(
                hybridization_ligand, self.num_hybridization
            ).float()

            hybridization_pocket = torch.zeros(
                pos_pocket.shape[0],
                hybridization_ligand.shape[1],
                dtype=torch.float32,
                device=device,
            )
            ligand_adds = {
                "hybridization": hybridization_ligand,
            }
            pocket_adds = {
                "hybridization": hybridization_pocket,
            }
            ligand_feats.update(ligand_adds)
            pocket_feats.update(pocket_adds)
        else:
            hybridization_ligand = None
            hybridization_pocket = None

        if inpainting:
            if self.hparams.additional_feats:
                raise ValueError("Inpainting not implemented for additional features")
            if hasattr(pocket_data, "pos_ligand"):
                pos_ligand_initial = pocket_data.pos_ligand.to(device)
            elif hasattr(pocket_data, "pos"):
                pos_ligand_initial = pocket_data.pos.to(device)
            else:
                raise ValueError("No ligand position provided for inpainting")
            # pos_ligand_initial is assumed to
            # lie within the center of mass of the pocket
            pos_ligand_initial = pos_ligand_initial - com[batch_ligand]

            lig_inpaint_mask = pocket_data.lig_inpaint_mask.to(device)
            lig_inpaint_mask_f = lig_inpaint_mask.float().unsqueeze(-1)
            atom_types_ligand_initial = pocket_data.x.to(device)
            # _ = pocket_data.numHs.to(device)
            if emd_ot:
                pos_ligand = self.optimal_transport_alignment(
                    pos_ligand_initial, pos_ligand, batch_ligand
                )

            if self.hparams.addHybridization:
                hybridization_ligand_initial = pocket_data.hybridization.to(device)
            else:
                hybridization_ligand_initial = None
            _ = hybridization_ligand_initial

        # fully-connected edge-index for ligand
        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)

        if inpainting:
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

        # initialize edge attributes
        (
            edge_attr_ligand,
            edge_index_ligand,
            mask,
            mask_i,
        ) = ut.initialize_edge_attrs_reverse(
            edge_index_ligand,
            n,
            self.bonds_prior,
            self.num_bond_classes,
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

        chain = range(0, self.T)
        chain = chain[::every_k_step]
        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )

        if clash_guidance_start is None:
            clash_guidance_start = 0
        if clash_guidance_end is None:
            clash_guidance_end = self.T

        for i, timestep in enumerate(iterator):

            for k in range(resample_steps):

                if self.node_level_t:  # or inpainting:
                    N = len(batch_ligand)
                else:
                    N = bs

                s = torch.full(
                    size=(N,), fill_value=timestep, dtype=torch.long, device=device
                )
                t = s + 1

                if self.node_level_t:  # or inpainting:
                    assert t.size(0) == batch_ligand.size(
                        0
                    ), "timesteps must be equal to the number of nodes for ligands"
                    tpocket = scatter_mean(t, batch_ligand, dim=0).long()
                    tpocket = tpocket[batch_pocket]
                    t_ = torch.concat([t, tpocket], dim=0)
                else:
                    t_ = t

                # time embedding
                temb = t_.float() / self.hparams.timesteps  # type: ignore [attr-defined]
                temb = temb.clamp(min=1e-3)
                temb = temb.unsqueeze(dim=1)

                node_feats_in = torch.cat(
                    [pl_feats_dict[feat] for feat in self.select_node_order], dim=-1
                )

                if self.hparams.additional_feats:
                    add_feats = torch.cat(
                        [pl_feats_dict[feat] for feat in self.select_add_order], dim=-1
                    )
                    node_feats_in = torch.cat([node_feats_in, add_feats], dim=-1)

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
                    pos=pl_feats_dict["pos"],
                    v=v,
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
                    variable_mask=(
                        fragment_mask
                        if fragment_mask is not None
                        else pl_feats_dict["ligand_mask"].unsqueeze(-1).float()
                    ),
                )

                pos_ligand_pred = out["coords_pred"].squeeze()
                atoms_pred = out["atoms_pred"]
                k = atoms_pred.size(-1)

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

                if self.hparams.noise_scheduler == "adaptive":  # type: ignore [attr-defined]
                    # positions
                    pos_ligand = self.sde_pos.sample_reverse_adaptive(
                        s,
                        t,
                        pos_ligand,
                        pos_ligand_pred,
                        batch_ligand,
                        cog_proj=False,
                        eta_ddim=eta_ddim,
                        probability_flow_ode=probability_flow_ode,
                        extend_t=True,
                    )  # here is cog_proj false as it will be downprojected later
                else:
                    # positions
                    pos_ligand = self.sde_pos.sample_reverse(
                        t,
                        pos_ligand,
                        pos_ligand_pred,
                        batch_ligand,
                        cog_proj=False,
                        eta_ddim=eta_ddim,
                        extend_t=True,
                    )  # here is cog_proj false as it will be downprojected later

                # atom elements
                atom_types_ligand = self.cat_atoms.sample_reverse_categorical(
                    xt=atom_types_ligand,
                    x0=atoms_ligand_pred,
                    t=t[batch_ligand],
                    num_classes=self.num_atom_types,
                )

                # charges
                charge_types_ligand = self.cat_charges.sample_reverse_categorical(
                    xt=charge_types_ligand,
                    x0=charges_ligand_pred,
                    t=t[batch_ligand],
                    num_classes=self.num_charge_classes,
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

                ligand_feats: Dict[str, Tensor] = {
                    "atom_types": atom_types_ligand,
                    "charges": charge_types_ligand,
                    "pos": pos_ligand,
                    "batch": batch_ligand,
                }

                if self.hparams.addNumHs:
                    # number of attached hydrogens
                    numHs_ligand = self.cat_numHs.sample_reverse_categorical(
                        xt=numHs_ligand,
                        x0=numHs_pred,
                        t=t[batch_ligand],
                        num_classes=self.num_Hs,
                    )
                    ligand_feats["numHs"] = numHs_ligand

                if self.hparams.addHybridization:
                    hybridization_ligand = (
                        self.cat_hybridization.sample_reverse_categorical(
                            xt=hybridization_ligand,
                            x0=hybridization_pred,
                            t=t[batch_ligand],
                            num_classes=self.num_hybridization,
                        )
                    )
                    ligand_adds = {
                        "hybridization": hybridization_ligand,
                    }
                    ligand_feats.update(ligand_adds)

                # edges
                (
                    edge_attr_ligand,
                    edge_index_ligand,
                    mask,
                    mask_i,
                ) = self.cat_bonds.sample_reverse_edges_categorical(
                    edge_attr_ligand,
                    edges_ligand_pred,
                    t,
                    mask,
                    mask_i,
                    batch=batch_ligand,
                    edge_index_global=edge_index_ligand,
                    num_classes=self.num_bond_classes,
                    extend_t=True,
                )

                if inpainting:
                    if self.node_level_t:
                        raise NotImplementedError(
                            "Inpainting not implemented for node-level t"
                        )
                    pos_ligand, atom_types_ligand, edge_attr_ligand = (
                        self._inpaint_forward_combine(
                            s=s,
                            pos_ligand_initial=pos_ligand_initial,
                            batch_ligand=batch_ligand,
                            atom_types_ligand_initial=atom_types_ligand_initial,
                            pos_ligand=pos_ligand,
                            lig_inpaint_mask_f=lig_inpaint_mask_f,
                            atom_types_ligand=atom_types_ligand,
                            edge_index_global_lig=edge_index_ligand,
                            edge_attr_ligand_initial=edge_attr_ligand_initial,
                            edge_mask_f=edge_mask_f,
                            edge_attr_ligand=edge_attr_ligand,
                            inpaint_edges=inpaint_edges,
                        )
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

                if inpainting and k < resample_steps - 1 and i < self.T - 1:
                    # forward noising q(x_{t} | x_{t-1})

                    # coords
                    s2 = self.sde_pos.discrete_betas[t]
                    mean = (1.0 - s2).sqrt()
                    sigma = s2.sqrt()
                    pos_ligand = pos_ligand * mean[batch_ligand].unsqueeze(
                        -1
                    ) + torch.randn_like(pos_ligand) * sigma[batch_ligand].unsqueeze(-1)

                    # atoms
                    _, atom_types_ligand = self.cat_atoms.sample_categorical(
                        t,
                        atom_types_ligand.argmax(dim=-1),
                        batch_ligand,
                        self.dataset_info,
                        num_classes=self.num_atom_types,
                        type="atoms",
                        cumulative=False,
                    )
                    # edges
                    edge_attr_ligand = self.cat_bonds.sample_edges_categorical(
                        t,
                        edge_index_ligand,
                        edge_attr_ligand.argmax(dim=-1),
                        batch_ligand,
                        return_one_hot=True,
                        cumulative=False,
                    )

            if save_traj:
                coords_traj.append(pos_ligand.detach().cpu().numpy())
                atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
                edges_traj.append(edge_attr_ligand.detach().cpu().numpy())

        # at last step, just infill the ground truth inpaintings
        if inpainting:
            last = True
            if last:
                pos_ligand, atom_types_ligand, edge_attr_ligand = (
                    self._inpaint_forward_combine(
                        s=s,
                        pos_ligand_initial=pos_ligand_initial,
                        batch_ligand=batch_ligand,
                        atom_types_ligand_initial=atom_types_ligand_initial,
                        pos_ligand=pos_ligand,
                        lig_inpaint_mask_f=lig_inpaint_mask_f,
                        atom_types_ligand=atom_types_ligand,
                        edge_index_global_lig=edge_index_ligand,
                        edge_attr_ligand_initial=edge_attr_ligand_initial,
                        edge_mask_f=edge_mask_f,
                        edge_attr_ligand=edge_attr_ligand,
                        inpaint_edges=inpaint_edges,
                    )
                )

        # Move generated molecule back to the original pocket position for docking
        pos_ligand += com[batch_ligand]
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
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            build_obabel_mol=build_obabel_mol,
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

    def reverse_sampling_node_level_t(
        self,
        pocket_data: Tensor,
        device: str,
        verbose: bool = False,
        every_k_step: int = 1,
        z: Optional[Tensor] = None,
        latent_gamma: float = 1.0,
        save_traj: bool = False,
        clash_guidance: bool = False,
        clash_guidance_scale: float = 0.01,
        clash_guidance_start: Optional[int] = None,
        clash_guidance_end: Optional[int] = None,
        T: Optional[int] = None,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        **kwargs,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:

        pos_pocket = pocket_data.pos_pocket.to(device)
        batch_pocket = pocket_data.pos_pocket_batch.to(device)
        x_pocket = pocket_data.x_pocket.to(device)

        batch_ligand = pocket_data.batch

        # bs = num_graphs
        bs = pocket_data.batch.max() + 1
        coords_traj = []
        atoms_traj = []
        edges_traj = []

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

            if self.old_conditional_learning:
                pos_pocket = pos_pocket - ref_points[batch_pocket]
                pos_ligand_initial = pos_ligand_initial - ref_points[batch_ligand]
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

        atom_types_ligand_initial = pocket_data.x.to(device)
        atom_types_ligand_initial = F.one_hot(
            atom_types_ligand_initial, self.num_atom_types
        ).float()

        if self.hparams.addNumHs:
            numHs_ligand = torch.multinomial(
                self.numHs_prior, num_samples=len(pos_ligand), replacement=True
            ).squeeze()
            numHs_ligand = F.one_hot(numHs_ligand, self.num_Hs).float()

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
        else:
            numHs_ligand = None
            numHs_ligand_initial = None
            numHs_pocket = None

        if self.hparams.addHybridization:
            hybridization_ligand = torch.multinomial(
                self.hybridization_prior, num_samples=len(pos_ligand), replacement=True
            ).squeeze()
            hybridization_ligand = F.one_hot(
                hybridization_ligand, self.num_hybridization
            ).float()

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
        else:
            hybridization_ligand = None
            hybridization_ligand_initial = None
            hybridization_pocket = None

        n = len(pos_ligand)

        # initialize atomic coordinates with variable parts translated to the COM of reference
        if not self.old_conditional_learning:
            pos_ligand = torch.randn_like(pos_ligand)
            pos_ligand = pos_ligand + ref_points[batch_ligand] * (
                1.0 - lig_inpaint_mask_f
            )

        # initialize the atom- and charge types
        atom_types_ligand = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types_ligand = F.one_hot(atom_types_ligand, self.num_atom_types).float()

        charge_types_ligand = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types_ligand = F.one_hot(
            charge_types_ligand, self.num_charge_classes
        ).float()

        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charge_types_pocket = torch.zeros(
            pos_pocket.shape[0],
            charge_types_ligand.shape[1],
            dtype=torch.float32,
            device=device,
        )

        # fully-connected edge-index for ligand
        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)

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

        # initialize edge attributes
        (
            edge_attr_ligand,
            edge_index_ligand,
            mask,
            mask_i,
        ) = ut.initialize_edge_attrs_reverse(
            edge_index_ligand,
            n,
            self.bonds_prior,
            self.num_bond_classes,
        )

        # infilling

        # combine
        pos_ligand = (
            pos_ligand * (1.0 - lig_inpaint_mask_f)
            + pos_ligand_initial * lig_inpaint_mask_f
        )
        atom_types_ligand = (
            atom_types_ligand * (1.0 - lig_inpaint_mask_f)
            + atom_types_ligand_initial * lig_inpaint_mask_f
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
            numHs_ligand = (
                numHs_ligand * (1.0 - lig_inpaint_mask_f)
                + numHs_ligand_initial * lig_inpaint_mask_f
            )
            ligand_feats["numHs"] = numHs_ligand
            pocket_feats["numHs"] = numHs_pocket

        if self.hparams.addHybridization:
            hybridization_ligand = (
                hybridization_ligand * (1.0 - lig_inpaint_mask_f)
                + hybridization_ligand_initial * lig_inpaint_mask_f
            )
            ligand_feats["hybridization"] = deepcopy(hybridization_ligand)
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

        chain = range(0, self.T) if T is None else range(0, T)
        chain = chain[::every_k_step]
        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )

        s_inpaint = (
            torch.ones_like(lig_inpaint_mask).long() * 1
        )  # discrete time steps, t=1 means data if T=max_num_timesteps

        if clash_guidance_start is None:
            clash_guidance_start = 0
        if clash_guidance_end is None:
            clash_guidance_end = self.T

        if save_traj:
            coords_traj.append(
                pos_ligand.detach().cpu().numpy()
                + com[batch_ligand].cpu().numpy()
                + (
                    ref_points[batch_ligand].cpu().numpy()
                    if self.old_conditional_learning
                    else 0.0
                )
            )
            atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
            edges_traj.append(edge_attr_ligand.detach().cpu().numpy())

        for i, timestep in enumerate(iterator):
            if self.node_level_t:
                N = len(batch_ligand)
            else:
                N = bs

            s = torch.full(
                size=(N,), fill_value=timestep, dtype=torch.long, device=device
            )
            s = s_inpaint * lig_inpaint_mask.long() + s * (1 - lig_inpaint_mask.long())
            t = s + 1

            assert t.size(0) == batch_ligand.size(
                0
            ), "timesteps must be equal to the number of nodes for ligands"
            tpocket = scatter_mean(t, batch_ligand, dim=0).long()
            tpocket = tpocket[batch_pocket]
            t_ = torch.concat([t, tpocket], dim=0)

            # time embedding
            temb = t_.float() / self.hparams.timesteps  # type: ignore [attr-defined]
            temb = temb.clamp(min=1e-3)
            temb = temb.unsqueeze(dim=1)

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

            pos_ligand = self.sde_pos.sample_reverse_adaptive(
                s,
                t,
                pos_ligand,
                pos_ligand_pred,
                batch_ligand,
                cog_proj=False,
                eta_ddim=1.0,
                probability_flow_ode=False,
                extend_t=False,
            )  # here is cog_proj false as it will be downprojected later

            atom_types_ligand = self.cat_atoms.sample_reverse_categorical(
                xt=atom_types_ligand,
                x0=atoms_ligand_pred,
                t=t,
                num_classes=self.num_atom_types,
            )

            # charges
            charge_types_ligand = self.cat_charges.sample_reverse_categorical(
                xt=charge_types_ligand,
                x0=charges_ligand_pred,
                t=t,
                num_classes=self.num_charge_classes,
            )

            if numHs_ligand is not None:
                numHs_ligand = self.cat_numHs.sample_reverse_categorical(
                    xt=numHs_ligand,
                    x0=numHs_pred,
                    t=t,
                    num_classes=self.num_Hs,
                )

            if hybridization_ligand is not None:
                hybridization_ligand = (
                    self.cat_hybridization.sample_reverse_categorical(
                        xt=hybridization_ligand,
                        x0=hybridization_pred,
                        t=t,
                        num_classes=self.num_hybridization,
                    )
                )

            # infilling
            # combine
            if pos_ligand_pred.isnan().any() or pos_ligand.isnan().any():
                # print("replacing nan with 0 at timestep", timestep)
                pos_ligand[pos_ligand.isnan()] = 0.0

            pos_ligand = (
                pos_ligand * (1.0 - lig_inpaint_mask_f)
                + pos_ligand_initial * lig_inpaint_mask_f
            )
            atom_types_ligand = (
                atom_types_ligand * (1.0 - lig_inpaint_mask_f)
                + atom_types_ligand_initial * lig_inpaint_mask_f
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

            ligand_feats: Dict[str, Tensor] = {
                "atom_types": atom_types_ligand,
                "charges": charge_types_ligand,
                "pos": pos_ligand,
                "batch": batch_ligand,
            }

            if self.hparams.addNumHs:
                # in-filling
                numHs_ligand = (
                    numHs_ligand * (1.0 - lig_inpaint_mask_f)
                    + numHs_ligand_initial * lig_inpaint_mask_f
                )
                ligand_feats["numHs"] = numHs_ligand
            if self.hparams.addHybridization:
                # in-filling
                hybridization_ligand = (
                    hybridization_ligand * (1.0 - lig_inpaint_mask_f)
                    + hybridization_ligand_initial * lig_inpaint_mask_f
                )
                ligand_feats["hybridization"] = hybridization_ligand

            # edges
            (
                edge_attr_ligand,
                edge_index_ligand,
                mask,
                mask_i,
            ) = self.cat_bonds.sample_reverse_edges_categorical(
                edge_attr_ligand,
                edges_ligand_pred,
                t,
                mask,
                mask_i,
                batch=batch_ligand,
                edge_index_global=edge_index_ligand,
                num_classes=self.num_bond_classes,
                extend_t=False,
            )

            edge_attr_ligand = (
                edge_attr_ligand * (1.0 - edge_mask_f)
                + edge_attr_ligand_initial * edge_mask_f
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

            if save_traj:
                coords_traj.append(
                    pos_ligand.detach().cpu().numpy()
                    + com[batch_ligand].cpu().numpy()
                    + (
                        ref_points[batch_ligand].cpu().numpy()
                        if self.old_conditional_learning
                        else 0.0
                    )
                )
                atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
                edges_traj.append(edge_attr_ligand.detach().cpu().numpy())

        # Move generated molecule back to the original pocket position for docking
        pos_ligand += com[batch_ligand]
        if self.old_conditional_learning:
            pos_ligand += ref_points[batch_ligand]
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

        if self.hparams.addHybridization:
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
            relax_mol=relax_mol,
            sanitize=sanitize,
            max_relax_iter=max_relax_iter,
            build_obabel_mol=build_obabel_mol,
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

        atom_types_prior = dataset_info.atom_types.float()
        charge_types_prior = dataset_info.charges_marginals.float()
        bond_types_prior = dataset_info.edge_types.float()
        self.register_buffer("atoms_prior", atom_types_prior.clone())
        self.register_buffer("bonds_prior", bond_types_prior.clone())
        self.register_buffer("charges_prior", charge_types_prior.clone())
        if self.hparams.addNumHs:
            numHs_prior = dataset_info.numHs.float()
            self.register_buffer("numHs_prior", numHs_prior.clone())

        if hparams["additional_feats"]:
            ring_prior = dataset_info.is_in_ring.float()
            self.register_buffer("ring_prior", ring_prior.clone())
            aromatic_prior = dataset_info.is_aromatic.float()
            self.register_buffer("aromatic_prior", aromatic_prior.clone())
            hybridization_prior = dataset_info.hybridization.float()
            self.register_buffer("hybridization_prior", hybridization_prior.clone())

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

        self.sde_pos = DiscreteDDPM(
            scaled_reverse_posterior_sigma=True,
            schedule=hparams["noise_scheduler"],
            nu=2.5,
            enforce_zero_terminal_snr=False,
            T=hparams["timesteps"],
            clamp_alpha_min=0.05,
        )
        self.sde_atom_charge = DiscreteDDPM(
            scaled_reverse_posterior_sigma=True,
            schedule=hparams["noise_scheduler"],
            nu=1.0,
            enforce_zero_terminal_snr=False,
            T=hparams["timesteps"],
            clamp_alpha_min=0.05,
        )
        self.cat_atoms = CategoricalDiffusionKernel(
            terminal_distribution=atom_types_prior,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_prior,
            alphas=self.sde_atom_charge.alphas.clone(),
        )
        if self.hparams.addNumHs:
            self.cat_numHs = CategoricalDiffusionKernel(
                terminal_distribution=numHs_prior,
                alphas=self.sde_atom_charge.alphas.clone(),
            )
        self.sde_bonds = DiscreteDDPM(
            scaled_reverse_posterior_sigma=True,
            schedule=hparams["noise_scheduler"],
            nu=1.5,
            enforce_zero_terminal_snr=False,
            T=hparams["timesteps"],
            clamp_alpha_min=0.05,
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_prior,
            alphas=self.sde_bonds.alphas.clone(),
        )

        if hparams["additional_feats"]:
            self.cat_ring = CategoricalDiffusionKernel(
                terminal_distribution=ring_prior,
                alphas=self.sde_atom_charge.alphas.clone(),
            )
            self.cat_aromatic = CategoricalDiffusionKernel(
                terminal_distribution=aromatic_prior,
                alphas=self.sde_atom_charge.alphas.clone(),
            )
            self.cat_hybridization = CategoricalDiffusionKernel(
                terminal_distribution=hybridization_prior,
                alphas=self.sde_atom_charge.alphas.clone(),
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

        atom_types_ligand: Tensor = batch.x
        pos_ligand: Tensor = batch.pos
        charges_ligand: Tensor = batch.charges
        batch_ligand: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        device = pos_ligand.device
        pos_ligand_com = scatter_mean(pos_ligand, batch_ligand, dim=0)
        pos_centered_ligand = pos_ligand - pos_ligand_com[batch_ligand]

        # forward noising for ligand features: coordinates, atom-types, charges
        _, pos_perturbed_ligand = self.sde_pos.sample_pos(
            t,
            pos_centered_ligand,
            batch_ligand,
            remove_mean=True,
            extend_t=not self.node_level_t,
            ot_alignment=self.hparams.ot_alignment,
        )
        atom_types_ligand, atom_types_ligand_perturbed = (
            self.cat_atoms.sample_categorical(
                t,
                atom_types_ligand,
                batch_ligand,
                self.dataset_info,
                num_classes=self.num_atom_types,
                type="atoms",
                extend_t=not self.node_level_t,
            )
        )
        charges_ligand, charges_ligand_perturbed = self.cat_charges.sample_categorical(
            t,
            charges_ligand,
            batch_ligand,
            self.dataset_info,
            num_classes=self.num_charge_classes,
            type="charges",
            extend_t=not self.node_level_t,
        )

        if self.hparams.additional_feats:
            ring_ligand: Tensor = batch.is_in_ring
            aromatic_ligand: Tensor = batch.is_aromatic
            hybridization_ligand: Tensor = batch.hybridization

            ring_ligand, ring_ligand_perturbed = self.cat_ring.sample_categorical(
                t,
                ring_ligand,
                batch_ligand,
                self.dataset_info,
                num_classes=self.num_is_in_ring,
                type="ring",
                extend_t=not self.node_level_t,
            )

            aromatic_ligand, aromatic_ligand_perturbed = (
                self.cat_aromatic.sample_categorical(
                    t,
                    aromatic_ligand,
                    batch_ligand,
                    self.dataset_info,
                    num_classes=self.num_is_aromatic,
                    type="aromatic",
                    extend_t=not self.node_level_t,
                )
            )

            hybridization_ligand, hybridization_ligand_perturbed = (
                self.cat_hybridization.sample_categorical(
                    t,
                    hybridization_ligand,
                    batch_ligand,
                    self.dataset_info,
                    num_classes=self.num_hybridization,
                    type="hybridization",
                    extend_t=not self.node_level_t,
                )
            )

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

        # forward noising edges
        edge_attr_global_perturbed_lig = self.cat_bonds.sample_edges_categorical(
            t,
            edge_index_global_lig,
            edge_attr_global_lig,
            batch_ligand,
            return_one_hot=True,
            extend_t=not self.node_level_t,
        )

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand_perturbed,
            "charges": charges_ligand_perturbed,
            "pos": pos_perturbed_ligand,
            "batch": batch_ligand,
        }

        if self.hparams.addNumHs:
            numHs_ligand: Tensor = batch.numHs
            numHs_ligand, numHs_ligand_perturbed = self.cat_numHs.sample_categorical(
                t,
                numHs_ligand,
                batch_ligand,
                self.dataset_info,
                num_classes=self.num_Hs,
                type="numHs",
                extend_t=not self.node_level_t,
            )
            ligand_feats["numHs"] = numHs_ligand_perturbed

        if self.hparams.additional_feats:
            ligand_adds = {
                "ring": ring_ligand_perturbed,
                "aromatic": aromatic_ligand_perturbed,
                "hybridization": hybridization_ligand_perturbed,
            }
            ligand_feats.update(ligand_adds)

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

        # time embedding
        temb = t.float() / self.hparams.timesteps  # type: ignore [attr-defined]
        temb = temb.clamp(min=1e-3)
        temb = temb.unsqueeze(dim=1)
        # forward pass for the model
        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
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
        if self.hparams.addNumHs:
            out["numHs_true"] = numHs_ligand.argmax(dim=-1)
        else:
            out["numHs_true"] = None
        out["bonds_true"] = edge_attr_global_lig
        out["charges_true"] = charges_ligand.argmax(dim=-1)
        out["bond_aggregation_index"] = edge_index_global_lig[1]

        if self.hparams.additional_feats:
            out["ring_true"] = ring_ligand.argmax(dim=-1)
            out["aromatic_true"] = aromatic_ligand.argmax(dim=-1)
            out["hybridization_true"] = hybridization_ligand.argmax(dim=-1)
        else:
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

    def _inpaint_forward_combine(
        self,
        s: Tensor,
        pos_ligand_initial: Tensor,
        batch_ligand: Tensor,
        atom_types_ligand_initial: Tensor,
        pos_ligand: Tensor,
        lig_inpaint_mask_f: Tensor,
        atom_types_ligand: Tensor,
        edge_index_global_lig: Tensor,
        edge_attr_ligand_initial: Tensor,
        edge_mask_f: Tensor,
        edge_attr_ligand: Tensor,
        inpaint_edges: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # inpainting through forward noising
        _, pos_ligand_inpaint = self.sde_pos.sample_pos(
            s,
            pos_ligand_initial,
            batch_ligand,
            remove_mean=False,
        )
        _, atom_types_inpaint = self.cat_atoms.sample_categorical(
            s,
            atom_types_ligand_initial,
            batch_ligand,
            self.dataset_info,
            num_classes=self.num_atom_types,
            type="atoms",
        )

        # combine
        pos_ligand = (
            pos_ligand * (1.0 - lig_inpaint_mask_f)
            + pos_ligand_inpaint * lig_inpaint_mask_f
        )
        atom_types_ligand = (
            atom_types_ligand * (1.0 - lig_inpaint_mask_f)
            + atom_types_inpaint * lig_inpaint_mask_f
        )

        if inpaint_edges:

            edge_attr_inpaint = self.cat_bonds.sample_edges_categorical(
                s,
                edge_index_global_lig,
                edge_attr_ligand_initial,
                batch_ligand,
                return_one_hot=True,
            )

            edge_attr_ligand = (
                edge_attr_ligand * (1.0 - edge_mask_f) + edge_attr_inpaint * edge_mask_f
            )

        return pos_ligand, atom_types_ligand, edge_attr_ligand

    def reverse_sampling(
        self,
        num_graphs: int,
        device: str,
        num_nodes_lig: Tensor,
        verbose: bool = False,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        build_obabel_mol=False,
        sa_importance_sampling=False,
        sa_importance_sampling_start=0,
        sa_importance_sampling_end=200,
        sa_every_importance_t=5,
        sa_tau=0.1,
        property_importance_sampling: bool = False,
        property_importance_sampling_start=0,
        property_importance_sampling_end=200,
        property_every_importance_t=5,
        property_tau: float = 0.1,
        joint_importance_sampling=False,
        property_normalization=False,
        latent_gamma: float = 1.0,
        clash_guidance: bool = False,
        clash_guidance_start=None,
        clash_guidance_end=None,
        clash_guidance_scale: float = 0.1,
        importance_gradient_guidance: float = 0.1,
        save_traj: bool = False,
        inpainting: bool = False,
        emd_ot: bool = False,
        z: Optional[Tensor] = None,
        _pos_ligand: Optional[Tensor] = None,
        probability_flow_ode: bool = False,
        inpaint_edges: bool = False,
        resample_steps: int = 1,
        resample_intervals: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:

        if joint_importance_sampling:
            print("joint importace sampling not implemented yet")
            raise ValueError

        batch_ligand = torch.arange(num_graphs, device=device).repeat_interleave(
            num_nodes_lig, dim=0
        )
        bs = num_graphs
        coords_traj = []
        atoms_traj = []
        edges_traj = []
        n = len(batch_ligand)
        pos_ligand = torch.randn(n, 3, device=device)
        pos_ligand = (
            pos_ligand - scatter_mean(pos_ligand, batch_ligand, dim=0)[batch_ligand]
        )

        # initialize the atom- and charge types
        atom_types_ligand = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types_ligand = F.one_hot(atom_types_ligand, self.num_atom_types).float()

        charge_types_ligand = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types_ligand = F.one_hot(
            charge_types_ligand, self.num_charge_classes
        ).float()

        ligand_feats: Dict[str, Tensor] = {
            "atom_types": atom_types_ligand,
            "charges": charge_types_ligand,
            "pos": pos_ligand,
            "batch": batch_ligand,
        }

        if self.hparams.addNumHs:
            numHs_ligand = torch.multinomial(
                self.numHs_prior, num_samples=n, replacement=True
            )
            numHs_ligand = F.one_hot(numHs_ligand, self.num_Hs).float()
            ligand_feats["numHs"] = numHs_ligand

        if self.hparams.additional_feats:
            ring_ligand = torch.multinomial(
                self.ring_prior, num_samples=n, replacement=True
            )
            ring_ligand = F.one_hot(ring_ligand, self.num_is_in_ring).float()

            aromatic_ligand = torch.multinomial(
                self.aromatic_prior, num_samples=n, replacement=True
            )
            aromatic_ligand = F.one_hot(aromatic_ligand, self.num_is_aromatic).float()

            hybridization_ligand = torch.multinomial(
                self.hybridization_prior, num_samples=n, replacement=True
            )
            hybridization_ligand = F.one_hot(
                hybridization_ligand, self.num_hybridization
            ).float()

            ligand_adds = {
                "ring": ring_ligand,
                "aromatic": aromatic_ligand,
                "hybridization": hybridization_ligand,
            }
            ligand_feats.update(ligand_adds)

        # fully-connected edge-index for ligand
        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)

        # initialize edge attributes
        (
            edge_attr_ligand,
            edge_index_ligand,
            mask,
            mask_i,
        ) = ut.initialize_edge_attrs_reverse(
            edge_index_ligand,
            n,
            self.bonds_prior,
            self.num_bond_classes,
        )

        # feature for interaction,
        # ligand-ligand
        edge_initial_interaction = torch.zeros(
            (edge_index_ligand.size(1), 3),
            dtype=torch.float32,
            device=device,
        )
        edge_initial_interaction[:, 0] = 1.0  # ligand-ligand
        batch_edge_global = batch_ligand[edge_index_ligand[0]]  #

        chain = range(0, self.T)
        chain = chain[::every_k_step]
        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )

        for timestep in iterator:

            for k in range(resample_steps):

                if self.node_level_t:
                    N = len(batch_ligand)
                else:
                    N = bs

                s = torch.full(
                    size=(N,), fill_value=timestep, dtype=torch.long, device=device
                )
                t = s + 1

                # time embedding
                temb = t.float() / self.hparams.timesteps  # type: ignore [attr-defined]
                temb = temb.clamp(min=1e-3)
                temb = temb.unsqueeze(dim=1)

                node_feats_in = torch.cat(
                    [ligand_feats[feat] for feat in self.select_node_order], dim=-1
                )

                if self.hparams.additional_feats:
                    add_feats = torch.cat(
                        [ligand_feats[feat] for feat in self.select_add_order], dim=-1
                    )
                    node_feats_in = torch.cat([node_feats_in, add_feats], dim=-1)

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

                if self.hparams.additional_feats:
                    k = atoms_pred.size(-1)
                    if self.hparams.addNumHs:
                        a, b, c = (
                            self.num_atom_types,
                            self.num_charge_classes,
                            self.num_Hs,
                        )
                        atoms_pred, charges_pred, numHs_pred, add_feats_pred = (
                            atoms_pred.split([a, b, c, k - a - b - c], dim=-1)
                        )
                        ring_pred, aromatic_pred, hybridization_pred = (
                            add_feats_pred.split(self.select_add_split, dim=-1)
                        )
                        ring_pred = ring_pred.softmax(dim=-1)
                        aromatic_pred = aromatic_pred.softmax(dim=-1)
                        hybridization_pred = hybridization_pred.softmax(dim=-1)
                        numHs_pred = numHs_pred.softmax(dim=-1)
                    else:
                        a, b = self.num_atom_types, self.num_charge_classes
                        atoms_pred, charges_pred, add_feats_pred = atoms_pred.split(
                            [a, b, k - a - b], dim=-1
                        )
                        ring_pred, aromatic_pred, hybridization_pred = (
                            add_feats_pred.split(self.select_add_split, dim=-1)
                        )
                        ring_pred = ring_pred.softmax(dim=-1)
                        aromatic_pred = aromatic_pred.softmax(dim=-1)
                        hybridization_pred = hybridization_pred.softmax(dim=-1)
                        numHs_pred = None
                else:
                    if self.hparams.addNumHs:
                        atoms_pred, charges_pred, numHs_pred = atoms_pred.split(
                            [self.num_atom_types, self.num_charge_classes, self.num_Hs],
                            dim=-1,
                        )
                        numHs_pred = numHs_pred.softmax(dim=-1)
                    else:
                        atoms_pred, charges_pred = atoms_pred.split(
                            [self.num_atom_types, self.num_charge_classes],
                            dim=-1,
                        )
                        numHs_pred = None
                    ring_pred = aromatic_pred = hybridization_pred = None

                atoms_ligand_pred = atoms_pred.softmax(dim=-1)
                # N x a_0
                edges_ligand_pred = out["bonds_pred"].softmax(dim=-1)
                # E x b_0
                charges_ligand_pred = charges_pred.softmax(dim=-1)

                # positions
                pos_ligand = self.sde_pos.sample_reverse_adaptive(
                    s,
                    # NOTE: needs adjustment for self.node_level_t = True,
                    # right now still on batch-size level
                    t,
                    # NOTE: needs adjustment for self.node_level_t = True,
                    # right now still on batch-size level
                    pos_ligand,
                    pos_ligand_pred,
                    batch_ligand,
                    cog_proj=False,
                    eta_ddim=eta_ddim,
                    probability_flow_ode=probability_flow_ode,
                    extend_t=True,
                    # NOTE: needs adjustment for self.node_level_t = True,
                    # right now still on batch-size level
                )  # here is cog_proj false as it will be downprojected later

                # atom elements
                atom_types_ligand = self.cat_atoms.sample_reverse_categorical(
                    xt=atom_types_ligand,
                    x0=atoms_ligand_pred,
                    t=t[batch_ligand],
                    # NOTE: currently inpainting case with self.node_level_t not covered!!
                    num_classes=self.num_atom_types,
                )

                # charges
                charge_types_ligand = self.cat_charges.sample_reverse_categorical(
                    xt=charge_types_ligand,
                    x0=charges_ligand_pred,
                    t=t[batch_ligand],
                    # NOTE: currently inpainting case with self.node_level_t not covered!!
                    num_classes=self.num_charge_classes,
                )

                ligand_feats: Dict[str, Tensor] = {
                    "atom_types": atom_types_ligand,
                    "charges": charge_types_ligand,
                    "pos": pos_ligand,
                    "batch": batch_ligand,
                }

                if self.hparams.addNumHs:
                    # number of attached hydrogens
                    numHs_ligand = self.cat_numHs.sample_reverse_categorical(
                        xt=numHs_ligand,
                        x0=numHs_pred,
                        t=t[batch_ligand],
                        num_classes=self.num_Hs,
                    )
                    ligand_feats["numHs"] = numHs_ligand

                if self.hparams.additional_feats:
                    ring_ligand = self.cat_ring.sample_reverse_categorical(
                        xt=ring_ligand,
                        x0=ring_pred,
                        t=t[batch_ligand],
                        num_classes=self.num_is_in_ring,
                    )
                    aromatic_ligand = self.cat_aromatic.sample_reverse_categorical(
                        xt=aromatic_ligand,
                        x0=aromatic_pred,
                        t=t[batch_ligand],
                        num_classes=self.num_is_aromatic,
                    )
                    hybridization_ligand = (
                        self.cat_hybridization.sample_reverse_categorical(
                            xt=hybridization_ligand,
                            x0=hybridization_pred,
                            t=t[batch_ligand],
                            num_classes=self.num_hybridization,
                        )
                    )
                    ligand_adds = {
                        "ring": ring_ligand,
                        "aromatic": aromatic_ligand,
                        "hybridization": hybridization_ligand,
                    }
                    ligand_feats.update(ligand_adds)

                # edges
                (
                    edge_attr_ligand,
                    edge_index_ligand,
                    mask,
                    mask_i,
                ) = self.cat_bonds.sample_reverse_edges_categorical(
                    edge_attr_ligand,
                    edges_ligand_pred,
                    t,
                    mask,
                    mask_i,
                    batch=batch_ligand,
                    edge_index_global=edge_index_ligand,
                    num_classes=self.num_bond_classes,
                    extend_t=True,
                    # NOTE: currently hardcoded and inpainting case not covered
                )

                if save_traj:
                    coords_traj.append(pos_ligand.detach().cpu().numpy())
                    atoms_traj.append(atom_types_ligand.detach().cpu().numpy())
                    edges_traj.append(edge_attr_ligand.detach().cpu().numpy())

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
                    "ring_pred": ring_ligand,
                    "aromatic_pred": aromatic_ligand,
                    "hybridization_pred": hybridization_ligand,
                }
            )

        molecules = get_generated_molecules(
            out_dict,
            batch_ligand,
            edge_index_ligand,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            num_Hs_classes=self.num_Hs if self.hparams.addNumHs else None,
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
        num_graphs: int,
        pocket_data: Tensor,
        device: str,
        verbose: bool = False,
        every_k_step: int = 1,
        z: Optional[Tensor] = None,
        latent_gamma: float = 1.0,
        save_traj: bool = False,
    ) -> Tuple[List[Molecule], Dict[str, Any]]:

        batch_ligand = pocket_data.batch

        bs = num_graphs
        coords_traj = []
        atoms_traj = []
        edges_traj = []

        if hasattr(pocket_data, "pos_ligand"):
            pos_ligand_initial = pocket_data.pos_ligand.to(device)
        elif hasattr(pocket_data, "pos"):
            pos_ligand_initial = pocket_data.pos.to(device)
        else:
            raise ValueError("No ligand position provided for inpainting")

        pos_ligand = pocket_data.pos.to(device)
        # initial features from the fixed ligand
        lig_inpaint_mask = pocket_data.lig_inpaint_mask.to(device)
        lig_inpaint_mask_f = lig_inpaint_mask.float().unsqueeze(-1)
        atom_types_ligand_initial = pocket_data.x.to(device)
        atom_types_ligand_initial = F.one_hot(
            atom_types_ligand_initial, self.num_atom_types
        ).float()

        n = len(pos_ligand)

        # initialize the atom- and charge types
        atom_types_ligand = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types_ligand = F.one_hot(atom_types_ligand, self.num_atom_types).float()

        charge_types_ligand = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types_ligand = F.one_hot(
            charge_types_ligand, self.num_charge_classes
        ).float()

        # fully-connected edge-index for ligand
        edge_index_ligand = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_ligand, _ = dense_to_sparse(edge_index_ligand)
        edge_index_ligand = sort_edge_index(edge_index_ligand, sort_by_row=False)

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

        # initialize edge attributes
        (
            edge_attr_ligand,
            edge_index_ligand,
            mask,
            mask_i,
        ) = ut.initialize_edge_attrs_reverse(
            edge_index_ligand,
            n,
            self.bonds_prior,
            self.num_bond_classes,
        )

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
            pos_ligand * (1.0 - lig_inpaint_mask_f)
            + pos_ligand_initial * lig_inpaint_mask_f
        )
        atom_types_ligand = (
            atom_types_ligand * (1.0 - lig_inpaint_mask_f)
            + atom_types_ligand_initial * lig_inpaint_mask_f
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

        chain = range(0, self.T)
        chain = chain[::every_k_step]
        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )

        t_inpaint = (
            torch.ones_like(lig_inpaint_mask).long() * 2
        )  # discrete time steps, t=1 means data if T=max_num_timesteps

        for timestep in iterator:

            if self.node_level_t:
                N = len(batch_ligand)
            else:
                N = bs

            s = torch.full(
                size=(N,), fill_value=timestep, dtype=torch.long, device=device
            )
            t = s + 1
            t = t_inpaint * lig_inpaint_mask.long() + t * (1 - lig_inpaint_mask.long())

            assert t.size(0) == batch_ligand.size(
                0
            ), "timesteps must be equal to the number of nodes for ligands"

            # time embedding
            temb = t.float() / self.hparams.timesteps  # type: ignore [attr-defined]
            temb = temb.clamp(min=1e-3)
            temb = temb.unsqueeze(dim=1)

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

            # import pdb; pdb.set_trace()
            pos_ligand = self.sde_pos.sample_reverse_adaptive(
                s,
                t,
                pos_ligand,
                pos_ligand_pred,
                batch_ligand,
                cog_proj=False,
                eta_ddim=1.0,
                probability_flow_ode=False,
                extend_t=False,
            )  # here is cog_proj false as it will be downprojected later
            # import pdb; pdb.set_trace()
            # atom elements
            # pdb.set_trace()
            atom_types_ligand = self.cat_atoms.sample_reverse_categorical(
                xt=atom_types_ligand,
                x0=atoms_ligand_pred,
                t=t,
                num_classes=self.num_atom_types,
            )

            # charges
            charge_types_ligand = self.cat_charges.sample_reverse_categorical(
                xt=charge_types_ligand,
                x0=charges_ligand_pred,
                t=t,
                num_classes=self.num_charge_classes,
            )

            # infilling
            # combine
            pos_ligand[pos_ligand.isnan()] = 0.0
            pos_ligand = (
                pos_ligand * (1.0 - lig_inpaint_mask_f)
                + pos_ligand_initial * lig_inpaint_mask_f
            )
            atom_types_ligand = (
                atom_types_ligand * (1.0 - lig_inpaint_mask_f)
                + atom_types_ligand_initial * lig_inpaint_mask_f
            )

            ligand_feats: Dict[str, Tensor] = {
                "atom_types": atom_types_ligand,
                "charges": charge_types_ligand,
                "pos": pos_ligand,
                "batch": batch_ligand,
            }

            # edges
            (
                edge_attr_ligand,
                edge_index_ligand,
                mask,
                mask_i,
            ) = self.cat_bonds.sample_reverse_edges_categorical(
                edge_attr_ligand,
                edges_ligand_pred,
                t,
                mask,
                mask_i,
                batch=batch_ligand,
                edge_index_global=edge_index_ligand,
                num_classes=self.num_bond_classes,
                extend_t=False,
            )

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
            num_Hs_classes=self.num_Hs if self.hparams.addNumHs else None,
            device=device,
            mol_device="cpu",
            relax_mol=None,
            max_relax_iter=100,
            sanitize=None,
            while_train=False,
            build_obabel_mol=None,
        )

        return molecules, {
            "coords": coords_traj,
            "atoms": atoms_traj,
            "edges": edges_traj,
        }

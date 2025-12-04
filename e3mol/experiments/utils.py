from copy import deepcopy
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import remove_self_loops, sort_edge_index
from torch_scatter import scatter_mean, scatter_sum
from torch_sparse import coalesce

from e3mol.experiments.data.datainfo import DatasetInfo
from e3mol.experiments.data.molecule import Molecule


def clamp_norm(
    x: torch.Tensor,
    maxnorm: float,
    dim=-1,
    eps: float = 1.0e-08,
) -> torch.Tensor:
    norm = x.norm(p=2, dim=dim, keepdim=True)
    mask = (norm < maxnorm).type_as(x)
    return mask * x + (1 - mask) * (x / norm.clamp_min(eps) * maxnorm)


def get_masked_com(pos, mask, batch, n_variable=None):
    if mask.ndim == 1:
        mask = mask.unsqueeze(-1)
    mask = mask.float()
    pos = pos * mask
    pos = scatter_sum(pos, batch, dim=0)
    n_masked = scatter_sum(mask, batch, dim=0)
    n_masked = n_masked.clamp(min=1)
    if n_variable is not None:
        n_masked = n_variable.unsqueeze(-1).float()
        n_masked = n_masked.clamp_min(1)
    pos = pos / n_masked
    return pos


def coalesce_edges(
    edge_index: Tensor, bond_edge_index: Tensor, bond_edge_attr: Tensor, n: int
) -> Tuple[Tensor, Tensor]:

    edge_attr = torch.full(
        size=(edge_index.size(-1),),
        fill_value=0,
        device=edge_index.device,
        dtype=torch.long,
    )
    edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
    edge_attr = torch.cat([edge_attr, bond_edge_attr], dim=0)
    edge_index, edge_attr = coalesce(
        index=edge_index, value=edge_attr, m=n, n=n, op="max"
    )
    return edge_index, edge_attr


def concat_ligand_pocket(
    ligand_feat_dict: Dict[str, Tensor],
    pocket_feat_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:

    assert len(ligand_feat_dict) == len(
        pocket_feat_dict
    ), f"Ligand and pocket feature dictionaries must have the same keys {list(ligand_feat_dict.keys())}, {list(pocket_feat_dict.keys())}."
    assert set(ligand_feat_dict.keys()) == set(pocket_feat_dict.keys())
    out_dict = {
        k: torch.cat([ligand_feat_dict[k], pocket_feat_dict[k]], dim=0)
        for k in ligand_feat_dict.keys()
    }
    batch_ligand = ligand_feat_dict["batch"]
    batch_pocket = pocket_feat_dict["batch"]
    mask_ligand = torch.cat(
        [
            torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
            torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        ],
        dim=0,
    )
    out_dict["ligand_mask"] = mask_ligand
    return out_dict


def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0) -> Tensor:
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out


def remove_mean_ligand(
    pos_ligand: Tensor, pos_pocket: Tensor, batch_ligand: Tensor, batch_pocket: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    # centers around the mean of the ligand coordinates
    mean = scatter_mean(pos_ligand, batch_ligand, dim=0)
    pos_ligand = pos_ligand - mean[batch_ligand]
    pos_pocket = pos_pocket - mean[batch_pocket]
    return pos_ligand, pos_pocket, mean


def remove_mean_pocket(
    pos_ligand: Tensor, pos_pocket: Tensor, batch_ligand: Tensor, batch_pocket: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    # centers around the mean of the pocket coordinates
    mean = scatter_mean(pos_pocket, batch_pocket, dim=0)
    pos_ligand = pos_ligand - mean[batch_ligand]
    pos_pocket = pos_pocket - mean[batch_pocket]
    return pos_ligand, pos_pocket, mean


def initialize_edge_attrs_reverse(
    edge_index: Tensor,
    n: int,
    bonds_prior: Tensor,
    num_bond_classes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    device = edge_index.device
    # edge types for FC graph
    j, i = edge_index
    mask = j < i
    mask_i = i[mask]
    mask_j = j[mask]
    nE = len(mask_i)
    edge_attr_triu = torch.multinomial(bonds_prior, num_samples=nE, replacement=True)

    j = torch.concat([mask_j, mask_i])
    i = torch.concat([mask_i, mask_j])
    edge_index = torch.stack([j, i], dim=0)
    edge_attr = torch.concat([edge_attr_triu, edge_attr_triu], dim=0)
    edge_index, edge_attr = sort_edge_index(
        edge_index=edge_index, edge_attr=edge_attr, sort_by_row=False
    )
    j, i = edge_index
    mask = j < i
    mask_i = i[mask]
    mask_j = j[mask]

    # some assert
    edge_attr_dense = torch.zeros(size=(n, n), device=device, dtype=torch.long)
    edge_attr_dense[edge_index[0], edge_index[1]] = edge_attr
    assert (edge_attr_dense - edge_attr_dense.T).sum().float() == 0.0

    edge_attr = F.one_hot(edge_attr, num_bond_classes).float()
    return edge_attr, edge_index, mask, mask_i


def get_ligand_pocket_edges(
    batch_lig: Tensor,
    batch_pocket: Tensor,
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    cutoff_p: float,
    cutoff_lp: float,
) -> Tensor:

    # ligand-ligand is fully-connected
    adj_ligand = batch_lig[:, None] == batch_lig[None, :]
    adj_pocket = batch_pocket[:, None] == batch_pocket[None, :]
    adj_cross = batch_lig[:, None] == batch_pocket[None, :]

    with torch.no_grad():
        D_pocket = torch.cdist(pos_pocket, pos_pocket)
        D_cross = torch.cdist(pos_ligand, pos_pocket)

    # pocket-pocket is not fully-connected
    # but selected based on distance cutoff
    adj_pocket = adj_pocket & (D_pocket <= cutoff_p)
    # ligand-pocket is not fully-connected
    # but selected based on distance cutoff
    adj_cross = adj_cross & (D_cross <= cutoff_lp)

    adj = torch.cat(
        (
            torch.cat((adj_ligand, adj_cross), dim=1),
            torch.cat((adj_cross.T, adj_pocket), dim=1),
        ),
        dim=0,
    )
    edges = torch.stack(torch.where(adj), dim=0)  # COO format (2, n_edges)
    return edges


def get_joint_edge_attrs(
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    edge_attr_ligand: Tensor,
    num_bond_classes: int,
    cutoff_p: float = 5.0,
    cutoff_lp: float = 5.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    assert num_bond_classes == 5
    device = edge_attr_ligand.device
    edge_index_global = get_ligand_pocket_edges(
        batch_ligand,
        batch_pocket,
        pos_ligand,
        pos_pocket,
        cutoff_p=cutoff_p,
        cutoff_lp=cutoff_lp,
    )
    edge_index_global = sort_edge_index(edge_index=edge_index_global, sort_by_row=False)
    edge_index_global, _ = remove_self_loops(edge_index_global)
    edge_attr_global = torch.zeros(
        (edge_index_global.size(1), num_bond_classes),
        dtype=torch.float32,
        device=device,
    )

    edge_mask_ligand = (edge_index_global[0] < len(batch_ligand)) & (
        edge_index_global[1] < len(batch_ligand)
    )
    edge_mask_pocket = (edge_index_global[0] >= len(batch_ligand)) & (
        edge_index_global[1] >= len(batch_ligand)
    )
    edge_attr_global[edge_mask_ligand] = edge_attr_ligand

    # placeholder no-bond information
    edge_attr_global[edge_mask_pocket] = (
        torch.tensor([0, 0, 0, 0, 1]).float().to(edge_attr_global.device)
    )

    batch_full = torch.cat([batch_ligand, batch_pocket])
    batch_edge_global = batch_full[edge_index_global[0]]  #

    edge_mask_ligand_pocket = (edge_index_global[0] < len(batch_ligand)) & (
        edge_index_global[1] >= len(batch_ligand)
    )
    edge_mask_pocket_ligand = (edge_index_global[0] >= len(batch_ligand)) & (
        edge_index_global[1] < len(batch_ligand)
    )

    # feature for interaction,
    # ligand-ligand, pocket-pocket, ligand-pocket, pocket-ligand
    edge_initial_interaction = torch.zeros(
        (edge_index_global.size(1), 3),
        dtype=torch.float32,
        device=device,
    )

    edge_initial_interaction[edge_mask_ligand] = (
        torch.tensor([1, 0, 0]).float().to(edge_attr_global.device)
    )  # ligand-ligand

    edge_initial_interaction[edge_mask_pocket] = (
        torch.tensor([0, 1, 0]).float().to(edge_attr_global.device)
    )  # pocket-pocket

    edge_initial_interaction[edge_mask_ligand_pocket] = (
        torch.tensor([0, 0, 1]).float().to(edge_attr_global.device)
    )  # ligand-pocket

    edge_initial_interaction[edge_mask_pocket_ligand] = (
        torch.tensor([0, 0, 1]).float().to(edge_attr_global.device)
    )  # pocket-ligand

    return (
        edge_index_global,
        edge_attr_global,
        batch_edge_global,
        edge_mask_ligand,
        edge_mask_pocket,
        edge_initial_interaction,
    )


def combine_protein_ligand_feats(
    ligand_feat_dict: Dict[str, Tensor],
    pocket_feat_dict: Dict[str, Tensor],
    edge_attr_ligand: Tensor,
    num_bond_classes: int,
    cutoff_p: float = 5.0,
    cutoff_lp: float = 5.0,
):
    """Wraps the utils.concat_ligand_pocket and utils.get_joint_edge_attrs
    into one function call
    """

    # get joint node-level features stacked as
    # [ligand, pocket] along the 0-th dimension
    pl_feats_dict = concat_ligand_pocket(
        ligand_feat_dict=ligand_feat_dict, pocket_feat_dict=pocket_feat_dict
    )

    # create protein-ligand complex edge-attrs
    (
        edge_index_global,
        edge_attr_global,
        batch_edge_global,
        edge_mask_ligand,
        _,
        edge_initial_interaction,
    ) = get_joint_edge_attrs(
        ligand_feat_dict["pos"],
        pocket_feat_dict["pos"],
        ligand_feat_dict["batch"],
        pocket_feat_dict["batch"],
        edge_attr_ligand,
        num_bond_classes,
        cutoff_p=cutoff_p,
        cutoff_lp=cutoff_lp,
    )

    out = (
        pl_feats_dict,
        edge_index_global,
        edge_attr_global,
        batch_edge_global,
        edge_mask_ligand,
        edge_initial_interaction,
    )

    return out


def get_fc_edge_index_with_offset(n: int, offset: int = 0, device="cpu"):
    row = torch.arange(n, dtype=torch.long)
    col = torch.arange(n, dtype=torch.long)
    row = row.view(-1, 1).repeat(1, n).view(-1)
    col = col.repeat(n)
    fc_edge_index = torch.stack([col, row], dim=0)
    mask = fc_edge_index[0] != fc_edge_index[1]
    fc_edge_index = fc_edge_index[:, mask]
    fc_edge_index += offset
    return fc_edge_index.to(device)


def get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes):
    ptr = torch.cat(
        [
            torch.zeros(1, device=batch_num_nodes.device, dtype=torch.long),
            batch_num_nodes.cumsum(0),
        ]
    )
    edge_tensor_lists = []
    for i in range(len(ptr) - 1):
        select_slice = slice(ptr[i].item(), ptr[i + 1].item())
        e = edge_attrs_dense[select_slice, select_slice]
        edge_tensor_lists.append(e)
    return edge_tensor_lists


def get_generated_molecules(
    out: dict,
    data_batch: Tensor,
    edge_index_global_lig: Tensor,
    dataset_info: DatasetInfo,
    device: str,
    relax_mol: bool = False,
    max_relax_iter: int = 200,
    sanitize: bool = False,
    mol_device: str = "cpu",
    check_validity: bool = False,
    build_obabel_mol: bool = False,
) -> List[Molecule]:

    atoms_pred = out["atoms_pred"]
    charges_pred = out["charges_pred"]

    batch_num_nodes = data_batch.bincount().cpu().tolist()
    pos_splits = (
        out["coords_pred"].detach().to(mol_device).split(batch_num_nodes, dim=0)
    )

    atom_types_integer = torch.argmax(atoms_pred, dim=-1).detach().to(mol_device)
    atom_types_integer_split = atom_types_integer.split(batch_num_nodes, dim=0)

    if charges_pred is not None:
        charge_types_integer = (
            torch.argmax(charges_pred, dim=-1).detach().to(mol_device)
        )
        # offset back
        charge_types_integer = charge_types_integer - dataset_info.charge_offset
        charge_types_integer_split = charge_types_integer.split(batch_num_nodes, dim=0)
    else:
        charge_types_integer_split = []

    if out["bonds_pred"] is not None:
        if out["bonds_pred"].shape[-1] > 5:
            out["bonds_pred"] = out["bonds_pred"][:, :5]
        n = data_batch.bincount().sum().item()
        edge_attrs_dense = torch.zeros(size=(n, n, 5), device=device).float()
        edge_attrs_dense[
            edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
        ] = out["bonds_pred"]
        edge_attrs_dense = edge_attrs_dense.argmax(-1).detach().to(mol_device)
        edge_attrs_splits = get_list_of_edge_adjs(
            edge_attrs_dense, data_batch.bincount()
        )
    else:
        edge_attrs_splits = []

    if "hybridization_pred" in out.keys():
        hybridization_feat = out["hybridization_pred"]
        hybridization_feat_integer = (
            torch.argmax(hybridization_feat, dim=-1).detach().to(mol_device)
        )
        hybridization_feat_integer_split = hybridization_feat_integer.split(
            batch_num_nodes, dim=0
        )
    else:
        hybridization_feat_integer_split = []

    if "numHs_pred" in out.keys():
        numHs_feat = out["numHs_pred"]
        numHs_feat_integer = torch.argmax(numHs_feat, dim=-1).detach().to(mol_device)
        numHs_feat_integer_split = numHs_feat_integer.split(batch_num_nodes, dim=0)
    else:
        numHs_feat_integer_split = []

    molecule_list = []

    for i, (
        positions,
        atom_types,
        charges,
        edges,
    ) in enumerate(
        zip_longest(
            pos_splits,
            atom_types_integer_split,
            charge_types_integer_split,
            edge_attrs_splits,
            fillvalue=None,
        )
    ):
        molecule = Molecule(
            atom_types=atom_types,
            positions=positions,
            charges=charges,
            bond_types=edges,
            hybridization=(
                hybridization_feat_integer_split[i]
                if len(hybridization_feat_integer_split) > 0
                else None
            ),
            num_Hs=(
                numHs_feat_integer_split[i]
                if len(numHs_feat_integer_split) > 0
                else None
            ),
            dataset_info=dataset_info,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            check_validity=check_validity,
            build_obabel_mol=build_obabel_mol,
        )
        molecule_list.append(molecule)

    return molecule_list


def create_copy_and_fill(
    data: Data,
    n_variable: int,
    lig_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> Data:

    n_total = len(lig_mask)
    n_fixed = data.pos.size(0)
    assert n_fixed <= n_total
    pos = data.pos
    x = data.x
    charge = data.charge

    if hasattr(data, "hybridization"):
        hybridization = data.hybridization
    else:
        hybridization = None

    if hasattr(data, "is_aromatic"):
        is_aromatic = data.is_aromatic
    else:
        is_aromatic = None

    data_mask = torch.zeros((n_total), device=pos.device, dtype=torch.bool)
    data_mask[:n_fixed] = True
    pos_new = torch.zeros((n_total, pos.size(1)), device=pos.device, dtype=pos.dtype)
    pos_mean = pos.mean(dim=0)
    pos_new[n_fixed:] = pos_mean
    x_new = torch.zeros((n_total,), device=x.device, dtype=x.dtype)
    charge_new = torch.zeros((n_total,), device=x.device, dtype=x.dtype)
    pos_new[:n_fixed, :] = pos
    x_new[:n_fixed] = x
    charge_new[:n_fixed] = charge

    if hybridization is not None:
        hybridization_new = torch.zeros((n_total,), device=x.device, dtype=x.dtype)
        hybridization_new[:n_fixed] = hybridization
    else:
        hybridization_new = None

    if is_aromatic is not None:
        is_aromatic_new = torch.zeros((n_total,), device=x.device, dtype=x.dtype)
        is_aromatic_new[:n_fixed] = is_aromatic
    else:
        is_aromatic_new = None

    data_copy = deepcopy(data)
    tmp = Data(
        x=x_new,
        pos=pos_new,
        charge=charge_new,
        is_aromatic=is_aromatic_new,
        hybridization=hybridization_new,
        lig_inpaint_mask=lig_mask,
        anchor_mask=anchor_mask,
        n_variable=n_variable,
        data_mask=data_mask,
    )
    tmp.charges = tmp.charge
    data_copy.update(tmp)

    return data_copy


def get_edge_mask_inpainting(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, fixed_nodes_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    if str(fixed_nodes_indices.dtype) == torch.bool:
        fixed_nodes_indices = torch.where(fixed_nodes_indices)[0]

    edge_0 = torch.where(edge_index[0][:, None] == fixed_nodes_indices[None, :])[0]

    edge_1 = torch.where(edge_index[1][:, None] == fixed_nodes_indices[None, :])[0]

    edge_index_between_fixed_nodes = edge_0[
        torch.where(edge_0[:, None] == edge_1[None, :])[0]
    ]

    edge_mask_between_fixed_nodes = torch.zeros_like(
        edge_attr, dtype=torch.bool, device=edge_index.device
    )

    edge_mask_between_fixed_nodes[edge_index_between_fixed_nodes] = True

    return edge_index_between_fixed_nodes, edge_mask_between_fixed_nodes


def prepare_inpainting_ligand_batch(
    data: Data,
    min_nodes_bias: int,
    max_nodes_bias: int,
    num_graphs: int,
    keep_ids: List | np.ndarray,
    anchor_idx: Optional[List[int]] = None,
) -> Batch:
    assert hasattr(data, "pos") or hasattr(
        data, "pos_ligand"
    ), "Data must have `pos` or `pos_ligand` attribute"
    assert hasattr(data, "x") or hasattr(
        data, "x_ligand"
    ), "Data must have `x` or `x_ligand` attribute"
    assert (
        hasattr(data, "charge")
        or hasattr(data, "charge_ligand")
        or hasattr(data, "charges")
    ), "Data must have `charge`, `charges`or `charge_ligand` attribute"

    if hasattr(data, "pos"):
        pos = data.pos
    elif hasattr(data, "pos_ligand"):
        pos = data.pos_ligand

    if hasattr(data, "x"):
        x = data.x
    elif hasattr(data, "x_ligand"):
        x = data.x_ligand

    if hasattr(data, "charge"):
        charge = data.charge
    elif hasattr(data, "charges"):
        charge = data.charges
    elif hasattr(data, "charge_ligand"):
        charge = data.charge_ligand

    data.pos = pos
    data.x = x
    data.charge = charge
    data.charges = charge
    device = data.pos.device

    assert 0 <= min_nodes_bias <= max_nodes_bias
    n_fixed = data.pos.size(0)

    if min_nodes_bias < max_nodes_bias:
        nodes_bias_ = torch.randint(
            low=min_nodes_bias,
            high=max_nodes_bias + 1,
            size=(num_graphs,),
            device=device,
            dtype=torch.long,
        )
    else:
        nodes_bias_ = torch.ones((num_graphs,), device=device, dtype=torch.long).fill_(
            min_nodes_bias
        )

    lig_mask_added = [
        torch.tensor([False] * n.item()).to(device).float() for n in nodes_bias_
    ]

    if keep_ids is None:
        lig_mask = torch.ones((n_fixed,), dtype=torch.bool, device=device)
    else:
        lig_mask = torch.zeros((n_fixed,), dtype=torch.bool, device=device)
        if isinstance(keep_ids, list):
            keep_ids = torch.tensor(keep_ids, device=device, dtype=torch.long)
        elif isinstance(keep_ids, np.ndarray):
            keep_ids = torch.from_numpy(keep_ids).to(device).long()
        lig_mask[keep_ids] = True

    anchor_mask = torch.zeros((n_fixed,), dtype=torch.bool, device=device)
    if anchor_idx is not None:
        anchor_mask[anchor_idx] = True

    n_variable = n_fixed - sum(lig_mask).cpu().item()

    lig_mask_batch = [
        torch.concat((lig_mask, added), dim=0) for added in lig_mask_added
    ]
    anchor_mask_batch = [
        torch.concat((anchor_mask, added), dim=0) for added in lig_mask_added
    ]

    datalist = [
        create_copy_and_fill(
            data,
            lig_mask=mask0,
            anchor_mask=mask1,
            n_variable=n_variable,
        )
        for mask0, mask1 in zip(lig_mask_batch, anchor_mask_batch)
    ]

    return Batch.from_data_list(datalist, follow_batch=["pos", "pos_pocket"])

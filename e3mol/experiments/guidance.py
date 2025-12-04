from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add, scatter_mean

from e3mol.experiments.utils import concat_ligand_pocket, get_fc_edge_index_with_offset
from e3mol.modules.model import DenoisingEdgeNetwork


def pocket_clash_guidance(
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    sigma: float = 2.0,
) -> Tuple[Tensor, Tensor]:
    with torch.enable_grad():
        x_in = pos_ligand.detach().requires_grad_(True)
        e = torch.exp(
            -torch.sum((pos_pocket.view(1, -1, 3) - x_in.view(-1, 1, 3)) ** 2, dim=-1)
            / float(sigma)
        )  # (n_l, n_p)
        connectivity_mask = batch_ligand.view(-1, 1) == batch_pocket.view(1, -1)
        e = e * connectivity_mask
        clash_loss = -sigma * torch.log(1e-3 + e.sum(dim=-1))  # (n_l,)
        clash_loss = scatter_mean(clash_loss, batch_ligand, dim=0)
        clash_loss = clash_loss.sum()  # (b,)->()
        grads = torch.autograd.grad(clash_loss, x_in)[0]
    return clash_loss, grads


def pseudo_lj_guidance(
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    dm_min: float = 0.5,
):
    dm_min = 0.5
    connectivity_mask = batch_ligand.view(-1, 1) == batch_pocket.view(1, -1)
    N = 6
    with torch.enable_grad():
        x_in = pos_ligand.detach().requires_grad_(True)
        dm = torch.sum(
            (pos_pocket.view(1, -1, 3) - x_in.view(-1, 1, 3)) ** 2, dim=-1
        ).sqrt()
        dm = torch.where(dm < dm_min, torch.ones_like(dm) * 1e10, dm)
        vdw_term1 = torch.pow(1 / dm, 2 * N)
        vdw_term2 = -2 * torch.pow(1 / dm, N)
        energy = vdw_term1 + vdw_term2
        energy = connectivity_mask * energy  # (n_l, n_p)
        energy = energy.sum(
            -1,
        )  # (n_l, )
        energy = scatter_add(energy, batch_ligand, dim=0).sum()
        grads = torch.autograd.grad(energy, x_in)[0]
    return energy, grads


def classifier_guidance_lp(
    model: DenoisingEdgeNetwork,
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    atom_types_ligand: Tensor,
    atom_types_pocket: Tensor,
    charge_types_ligand: Tensor,
    charge_types_pocket: Tensor,
    temb: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    edge_index_ligand: Tensor,
    edge_mask_ligand: Tensor,
    edge_initial_interaction: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    batch_full: Tensor,
    batch_edge_global: Tensor,
    guidance_scale=1e-1,
    minimize_property=False,
    z: OptTensor = None,
    latent_gamma: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:

    pos_ligand = pos_ligand.detach()
    pos_ligand.requires_grad = True
    pos_pocket = pos_pocket.detach()
    pos_pocket.requires_grad = False

    with torch.enable_grad():
        (
            pos_joint,
            atom_types_joint,
            charge_types_joint,
            batch_full,
            ligand_mask,
        ) = concat_ligand_pocket(
            pos_ligand,
            pos_pocket,
            atom_types_ligand,
            atom_types_pocket,
            charge_types_ligand,
            charge_types_pocket,
            batch_ligand,
            batch_pocket,
        )

        node_feats_in = torch.cat([atom_types_joint, charge_types_joint], dim=-1)

        out = model(
            x=node_feats_in,
            t=temb,
            pos=pos_joint,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_full,
            edge_attr_initial_ohe=edge_initial_interaction,
            edge_index_ligand=edge_index_ligand,
            batch_edge_global=batch_edge_global,
            z=z,
            ligand_mask=ligand_mask.unsqueeze(1),
            batch_ligand=batch_ligand,
            latent_gamma=latent_gamma,
            edge_mask_ligand=edge_mask_ligand,
        )
        if minimize_property:
            sign = -1.0
        else:
            sign = 1.0

        # docking_score
        if minimize_property:
            property_pred = out["property_pred"][1]
        else:
            property_pred = out["property_pred"][0]

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(property_pred)]
        grad_shift = torch.autograd.grad(
            [property_pred],
            [pos_ligand],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

    pos_ligand = pos_ligand + sign * guidance_scale * grad_shift[:, :3]
    pos_ligand.detach_()

    return pos_ligand, atom_types_ligand, charge_types_ligand


@torch.no_grad()
def importance_sampling(
    model: DenoisingEdgeNetwork,
    node_feats_in: Tensor,
    temb: Tensor,
    pos: Tensor,
    edge_index_global: Tensor,
    edge_attr_global: Tensor,
    batch: Tensor,
    batch_ligand: Tensor,
    batch_edge_global: Tensor,
    batch_num_nodes: Tensor,
    edge_index_ligand: Tensor,
    edge_attr_ligand: Tensor,
    ligand_mask: Tensor,
    edge_mask_ligand: Tensor,
    maximize_score: bool = True,
    sa_tau: float = 0.1,
    property_tau: float = 0.1,
    kind: str = "sa_score",
    property_normalization=False,
    edge_attr_initial_ohe=None,
    z=None,
    latent_gamma=1.0,
    guidance_scale=0.1,
    importance_gradient_guidance=False,
):
    """
    Idea:
    The point clouds / graphs have an intermediate predicted synthesizability.
    Given a set/population of B graphs/point clouds we want to __bias__
    the sampling process towards "regions"
    where the fitness (here the synth.) is maximized.
    Hence we can compute importance weights for
    each sample i=1,2,...,B and draw a new population with replacement.
    As the sampling process is stochastic, repeated samples will evolve differently.
    However we need to think about ways to also
    include/enforce uniformity such that some samples are not drawn too often.
    To make it more "uniform", we can use temperature annealing in the softmax
    """
    assert kind in ["sa_score", "docking_score", "ic50", "joint"]
    device = node_feats_in.device

    if not importance_gradient_guidance:
        out = model(
            x=node_feats_in,
            t=temb,
            pos=pos,
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            batch=batch,
            edge_attr_initial_ohe=edge_attr_initial_ohe,
            edge_index_ligand=edge_index_ligand,
            batch_edge_global=batch_edge_global,
            z=z,
            ligand_mask=ligand_mask.unsqueeze(1),
            batch_ligand=batch_ligand,
            latent_gamma=latent_gamma,
            edge_mask_ligand=edge_mask_ligand,
        )
    else:
        # gradient guidance
        ligand_mask = ligand_mask.bool()
        pos = pos.detach()
        pos_ligand = pos[ligand_mask].detach()
        pos_pocket = pos[~ligand_mask].detach()
        pos_pocket.requires_grad = False
        pos_ligand.requires_grad = True
        with torch.enable_grad():

            pos = torch.cat([pos_ligand, pos_pocket], dim=0)

            out = model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index=edge_index_global,
                edge_attr=edge_attr_global,
                batch=batch,
                edge_attr_initial_ohe=edge_attr_initial_ohe,
                edge_index_ligand=edge_index_ligand,
                batch_edge_global=batch_edge_global,
                z=z,
                ligand_mask=ligand_mask.unsqueeze(1),
                batch_ligand=batch_ligand,
                latent_gamma=latent_gamma,
                edge_mask_ligand=edge_mask_ligand,
            )

        if kind == "sa_score":
            property_pred = out["property_pred"][0]
            sign = 1.0
        elif kind == "docking_score":
            property_pred = out["property_pred"][1]
            sign = -1.0
        elif kind == "ic50":
            property_pred = out["property_pred"][1]
            sign = 1.0

        grad_outputs = [torch.ones_like(property_pred)]
        grad_shift = torch.autograd.grad(
            [property_pred],
            [pos_ligand],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

        pos_ligand = pos_ligand + sign * guidance_scale * grad_shift[:, :3]
        pos_ligand.detach_()
        pos = torch.cat([pos_ligand, pos_pocket], dim=0)

    # select features from ligands only in the PL complex
    ligand_mask = ligand_mask.bool()
    node_feats_in = node_feats_in[ligand_mask]
    pos = pos[ligand_mask]
    prop_pred = out["property_pred"]
    sa, prop = prop_pred
    if isinstance(sa, torch.Tensor):
        sa = sa.detach()
    if isinstance(prop, torch.Tensor):
        prop = prop.detach()
    sa = (
        sa.squeeze(1).sigmoid()
        if sa is not None and (kind == "sa_score" or kind == "joint")
        else None
    )
    if prop is not None and (
        kind == "docking_score" or kind == "ic50" or kind == "joint"
    ):
        if prop.dim() == 2:
            prop = prop.squeeze()
        if kind == "docking_score":
            prop = -1.0 * prop
            if property_normalization:
                N = batch_ligand.bincount().float()
                prop = prop / torch.sqrt(N)
    else:
        prop = None

    if not maximize_score and sa is not None:
        sa = 1.0 - sa

    n = pos.size(0)
    b = len(batch_num_nodes)

    weights_sa = (sa / sa_tau).softmax(dim=0) if sa is not None else None

    weights_prop = (prop / property_tau).softmax(dim=0) if prop is not None else None

    if kind == "joint":
        assert sa is not None and prop is not None
        # weights_add = 1.0 * (weights_sa + weights_prop)
        weights_add = weights_prop  # 0.0
        weights_mul = weights_sa * weights_prop
        weights = 1.0 * (weights_add + weights_mul)
        weights = weights.softmax(dim=0)
    elif kind == "sa_score":
        assert sa is not None
        weights = weights_sa
    elif kind == "docking_score" or kind == "ic50":
        assert prop is not None
        weights = weights_prop

    select = torch.multinomial(weights, num_samples=len(weights), replacement=True)
    select = select.sort()[0]
    ptr = torch.concat(
        [
            torch.zeros((1,), device=batch_num_nodes.device, dtype=torch.long),
            batch_num_nodes.cumsum(0),
        ],
        dim=0,
    )
    batch_num_nodes_new = batch_num_nodes[select]
    # select
    batch_new = torch.arange(b, device=pos.device).repeat_interleave(
        batch_num_nodes_new
    )
    # node level
    a, b = node_feats_in.size(1), pos.size(1)
    x = torch.concat([node_feats_in, pos], dim=1)
    x_split = x.split(batch_num_nodes.cpu().numpy().tolist(), dim=0)
    x_select = torch.concat([x_split[i] for i in select.cpu().numpy()], dim=0)
    node_feats_in, pos = x_select.split([a, b], dim=-1)
    # edge level
    edge_slices = [slice(ptr[i - 1].item(), ptr[i].item()) for i in range(1, len(ptr))]
    edge_slices_new = [edge_slices[i] for i in select.cpu().numpy()]

    # populate the dense edge-tensor
    E_dense = torch.zeros(
        (n, n, edge_attr_ligand.size(1)),
        dtype=edge_attr_ligand.dtype,
        device=edge_attr_ligand.device,
    )
    E_dense[edge_index_ligand[0], edge_index_ligand[1], :] = edge_attr_ligand

    # select the slices
    E_s = torch.stack(
        [
            torch.block_diag(*[E_dense[s, s, i] for s in edge_slices_new])
            for i in range(E_dense.size(-1))
        ],
        dim=-1,
    )
    new_ptr = torch.concat(
        [
            torch.zeros((1,), device=batch_num_nodes_new.device, dtype=torch.long),
            batch_num_nodes_new.cumsum(0),
        ],
        dim=0,
    )

    new_fc_edge_index = torch.concat(
        [
            get_fc_edge_index_with_offset(
                n=batch_num_nodes_new[i].item(), offset=new_ptr[i].item()
            )
            for i in range(len(new_ptr) - 1)
        ],
        dim=1,
    )

    new_edge_attr = E_s[new_fc_edge_index[0], new_fc_edge_index[1], :]

    out = (
        pos.to(device),
        node_feats_in.to(device),
        new_fc_edge_index.to(device),
        new_edge_attr.to(device),
        batch_new.to(device),
        {"batch_num_nodes_old": batch_num_nodes, "select": select},
        batch_num_nodes_new.to(device),
    )

    return out

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.typing import OptTensor

from e3mol.modules.convs import EQGATGlobalEdgeConvFinal, EQGATLocalConvFinal
from e3mol.modules.nn import AdaptiveLayerNorm, LayerNorm


class EQGATEdgeGNN(nn.Module):
    """_summary_
    EQGAT GNN Network
    updating node-level scalar, vectors and position features as well as edge-features.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hn_dim: Tuple[int, int] = (64, 16),
        edge_dim: int = 16,
        cutoff_local: float = 5.0,
        num_layers: int = 5,
        latent_dim: int = 0,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        use_out_norm: bool = True,
        store_intermediate_coords: bool = False,
        include_field_repr: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cutoff_local = cutoff_local
        self.store_intermediate_coords = store_intermediate_coords

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim
        convs = []

        for i in range(num_layers):
            convs.append(
                EQGATGlobalEdgeConvFinal(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    has_v_in=i > 0 or include_field_repr,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                    cutoff=cutoff_local,
                )
            )

        self.convs = nn.ModuleList(convs)

        if latent_dim:
            self.norms = nn.ModuleList(
                [
                    AdaptiveLayerNorm(dims=hn_dim, latent_dim=latent_dim)
                    for _ in range(num_layers)
                ]
            )
            self.out_norm = (
                AdaptiveLayerNorm(dims=hn_dim, latent_dim=latent_dim)
                if use_out_norm
                else None
            )
        else:
            self.norms = nn.ModuleList(
                [
                    LayerNorm(dims=hn_dim, latent_dim=latent_dim)
                    for _ in range(num_layers)
                ]
            )
            self.out_norm = (
                LayerNorm(dims=hn_dim, latent_dim=latent_dim) if use_out_norm else None
            )

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        if self.out_norm is not None:
            self.out_norm.reset_parameters()

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: OptTensor,
        pos: Tensor,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos_norm = torch.norm(pos, dim=1).unsqueeze(1)
        pos_n = torch.full_like(pos, fill_value=0.0)
        mask = (pos_norm != 0.0).squeeze(1)
        pos_n[mask] = pos[mask] / pos_norm[mask]
        a = pos_n[target] * pos_n[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        p: Tensor,
        batch: Tensor,
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        edge_attr_initial_ohe: Tensor,
        edge_attr_global_embedding: Tensor,
        z: OptTensor = None,
        ligand_mask: OptTensor = None,
        latent_gamma: float = 1.0,
    ):
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)

        pos_list = []

        edge_attr_global = edge_attr

        for i in range(len(self.convs)):
            edge_index_in = edge_index
            edge_attr_in = edge_attr_global
            s, v = self.norms[i](
                x={"s": s, "v": v, "z": z}, batch=batch, gamma=latent_gamma
            )
            out = self.convs[i](
                x=(s, v, p),
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
                ligand_mask=ligand_mask,
                edge_attr_initial_ohe=edge_attr_initial_ohe,
                edge_attr_global_embedding=edge_attr_global_embedding,
            )

            s, v, p, e = out["s"], out["v"], out["p"], out["e"]

            edge_attr_global = self.calculate_edge_attrs(
                edge_index=edge_index,
                pos=p,
                edge_attr=e,
            )

            if self.store_intermediate_coords and self.training:
                if i < len(self.convs) - 1:
                    if ligand_mask is not None:
                        pos_list.append(p[ligand_mask.squeeze(), :])
                    else:
                        pos_list.append(p)
                    p = p.detach()
            e = edge_attr_global[-1]

        if self.out_norm is not None:
            s, v = self.out_norm(
                x={"s": s, "v": v, "z": z}, batch=batch, gamma=latent_gamma
            )

        out = {
            "s": s,
            "v": v,
            "e": e,
            "p": p,
            "p_list": pos_list,
        }

        return out


class EQGATLocalGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and potentially coordinates.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hn_dim: Tuple[int, int] = (64, 16),
        edge_dim: int = 16,
        cutoff_local: float = 5.0,
        num_layers: int = 5,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        store_intermediate_coords: bool = False,
        use_out_norm: bool = True,
        coords_update: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cutoff_local = cutoff_local

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim

        convs = []
        self.store_intermediate_coords = store_intermediate_coords
        self.coords_update = coords_update
        for i in range(num_layers):
            convs.append(
                EQGATLocalConvFinal(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                    coords_update=coords_update,
                )
            )

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([LayerNorm(dims=hn_dim) for _ in range(num_layers)])

        self.out_norm = LayerNorm(dims=hn_dim) if use_out_norm else None

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        if self.out_norm is not None:
            self.out_norm.reset_parameters()

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        p: Tensor,
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch: Tensor = None,
        ligand_mask: Optional[Tensor] = None,
    ):
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)

        pos_list: list = []

        for i in range(len(self.convs)):
            s, v = self.norms[i](x={"s": s, "v": v}, batch=batch)
            out = self.convs[i](
                x=(s, v, p),
                edge_index=edge_index,
                edge_attr=edge_attr,
                ligand_mask=ligand_mask,
            )
            s, v, p = out["s"], out["v"], out["p"]
            # here edge-attributes are (currently) not recomputed, even if coords-update is True

            if self.store_intermediate_coords:
                pos_list.append(p)

        if self.out_norm is not None:
            s, v = self.out_norm(x={"s": s, "v": v}, batch=batch)
        out = {
            "s": s,
            "v": v,
            "p": p if self.coords_update else None,
            "p_list": pos_list,
        }

        return out

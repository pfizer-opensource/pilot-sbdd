import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_mean

from e3mol.modules.gnn import EQGATEdgeGNN, EQGATLocalGNN
from e3mol.modules.nn import DenseLayer


class PredictionHeadEdge(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        edge_dim: int,
        num_atom_features: int,
        num_bond_types: int = 5,
        model_score: bool = False,
    ) -> None:
        super().__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_features = num_atom_features

        self.shared_mapping = DenseLayer(
            self.sdim, self.sdim, bias=True, activation=nn.SiLU()
        )

        self.bond_mapping = DenseLayer(edge_dim, self.sdim, bias=True)

        self.bonds_lin_0 = DenseLayer(
            in_features=self.sdim + 1, out_features=self.sdim, bias=True
        )
        self.bonds_lin_1 = DenseLayer(
            in_features=self.sdim, out_features=num_bond_types, bias=True
        )

        self.v_out_features = 1 + int(model_score)
        self.coords_lin = DenseLayer(
            in_features=self.vdim, out_features=self.v_out_features, bias=False
        )
        self.atoms_lin = DenseLayer(
            in_features=self.sdim, out_features=num_atom_features, bias=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()

    def forward(
        self,
        x: Dict,
        batch: Tensor,
        edge_index: Tensor,
        edge_index_ligand: OptTensor = None,
        batch_ligand: OptTensor = None,
        ligand_mask: OptTensor = None,
        edge_mask_ligand: OptTensor = None,
    ):

        s, v, p, e = x["s"], x["v"], x["p"], x["e"]
        s = self.shared_mapping(s)
        coords_pred = self.coords_lin(v).squeeze()
        if self.v_out_features == 2:
            coords_pred, score_pred = coords_pred.split(1, dim=-1)
            coords_pred = coords_pred.squeeze()
            score_pred = score_pred.squeeze()
        else:
            score_pred = None

        atoms_pred = self.atoms_lin(s)
        if batch_ligand is not None and ligand_mask is not None:
            # selecting from the PL-complex the Ligand features
            s = (s * ligand_mask)[ligand_mask.squeeze(), :]
            j, i = edge_index_ligand
            atoms_pred = (atoms_pred * ligand_mask)[ligand_mask.squeeze(), :]
            coords_pred = (coords_pred * ligand_mask)[ligand_mask.squeeze(), :]
            if score_pred is not None:
                score_pred = (score_pred * ligand_mask)[ligand_mask.squeeze(), :]
            p = (p * ligand_mask)[ligand_mask.squeeze(), :]
            coords_pred = p + coords_pred
            d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)
        else:
            j, i = edge_index
            n = s.size(0)
            coords_pred = p + coords_pred
            coords_pred = (
                coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]
            )
            d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)

        if edge_mask_ligand is not None and edge_index_ligand is not None:
            # selecting from the PL-complex the Ligand features
            n = len(batch_ligand)
            e = (e * edge_mask_ligand.unsqueeze(1))[edge_mask_ligand]
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index_ligand[0], edge_index_ligand[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index_ligand[0], edge_index_ligand[1], :]
        else:
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index[0], edge_index[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index[0], edge_index[1], :]

        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)

        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)

        outs = (coords_pred, atoms_pred, bonds_pred, score_pred)
        return outs


class DenoisingEdgeNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction,
    i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        latent_dim: int = 0,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        use_out_norm: bool = True,
        amino_acid_ohe_transform: bool = False,
        store_intermediate_coords: bool = False,
        joint_property_prediction: bool = False,
        regression_property: Optional[List[str]] = None,
        node_level_t: bool = False,
        model_score: bool = False,
        include_field_repr: bool = False,
    ) -> None:
        super().__init__()
        if regression_property is None:
            regression_property = []
        self.joint_property_prediction = joint_property_prediction
        self.regression_property = regression_property
        self.num_bond_types = num_bond_types
        self.node_level_t = node_level_t

        self.store_intermediate_coords = store_intermediate_coords
        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim)

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping:
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        if amino_acid_ohe_transform:
            self.amino_acid_ohe_mapping = DenseLayer(20, hn_dim[0], bias=True)
        else:
            self.amino_acid_ohe_mapping = None  # type: ignore

        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)
        self.context_mapping = False

        self.sdim, self.vdim = hn_dim

        self.edge_pre = nn.Sequential(
            DenseLayer(60, 2 * 60, activation=nn.Softplus()),
            DenseLayer(2 * 60, 1, activation=nn.Sigmoid()),
        )

        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            edge_dim=edge_dim,
            cutoff_local=cutoff_local,
            latent_dim=latent_dim,
            num_layers=num_layers,
            use_cross_product=False,
            vector_aggr=vector_aggr,
            use_out_norm=use_out_norm,
            store_intermediate_coords=store_intermediate_coords,
            include_field_repr=include_field_repr,
        )

        self.prediction_head = PredictionHeadEdge(
            hn_dim=hn_dim,
            edge_dim=edge_dim,
            num_atom_features=num_atom_features,
            num_bond_types=num_bond_types,
            model_score=model_score,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.time_mapping_atom.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        if self.context_mapping and hasattr(
            self.atom_context_mapping, "reset_parameters"
        ):
            self.atom_context_mapping.reset_parameters()
        if self.context_mapping and hasattr(self.context_mapping, "reset_parameters"):
            self.context_mapping.reset_parameters()
        self.time_mapping_bond.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()

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
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        edge_attr_initial_ohe: Tensor,
        v: OptTensor = None,
        amino_acid_ohe: OptTensor = None,
        edge_index_ligand: OptTensor = None,
        batch_edge_global: OptTensor = None,
        z: OptTensor = None,
        ligand_mask: OptTensor = None,
        batch_ligand: OptTensor = None,
        latent_gamma: float = 1.0,
        edge_mask_ligand: OptTensor = None,
        anchor_fragment_embedding: OptTensor = None,
        variable_mask: OptTensor = None,
    ):

        bs = len(batch.unique())
        if ligand_mask is None:
            pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]

        if isinstance(ligand_mask, Tensor):
            if ligand_mask.dim() == 1:
                ligand_mask = ligand_mask.unsqueeze(-1)

        ta = self.time_mapping_atom(t)
        # t: (batch_size, 1)
        if len(t) == bs:
            tnode = ta[batch]
        else:
            # t already on node-level, i.e. t: (N, 1)
            tnode = ta

        if len(t) == bs:
            # edge_index_global (2, E*)
            tb = self.time_mapping_bond(t)
            tedge_global = tb[batch_edge_global]
        else:
            # t: (N, 1)
            t = (t + t.T) / 2.0
            # (N, N)
            j, i = edge_index
            t = t[j, i]
            tedge_global = self.time_mapping_bond(t.unsqueeze(-1))

        s = self.atom_mapping(x)
        s = self.atom_time_mapping(s + tnode)
        if anchor_fragment_embedding is not None:
            s = s + anchor_fragment_embedding

        if self.amino_acid_ohe_mapping is not None:
            assert isinstance(amino_acid_ohe, Tensor)
            s = self.amino_acid_ohe_mapping(amino_acid_ohe) + s

        edge_attr_global_transformed = self.bond_mapping(edge_attr)
        edge_attr_global_transformed = self.bond_time_mapping(
            edge_attr_global_transformed + tedge_global
        )
        if v is None:
            v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)
        else:
            assert isinstance(v, Tensor), "v should be a Tensor"
            assert v.ndim == 3, "v should have 3 dimensions"
            assert v.size(0) == x.size(0), "v should have the correct dimension"
            assert v.size(1) == 3, "v should have the correct dimension"
            assert v.size(2) == self.vdim, "v should have the correct dimension"

        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(
            edge_index=edge_index,
            edge_attr=edge_attr_global_transformed,
            pos=pos,
        )

        assert edge_attr_initial_ohe is not None
        assert edge_attr_initial_ohe.size(1) == 3
        d = edge_attr_global_transformed[0]
        rbf = self.gnn.convs[0].radial_basis_func(d)
        rbf_ohe = torch.einsum("nk, nd -> nkd", (rbf, edge_attr_initial_ohe))
        rbf_ohe = rbf_ohe.view(d.size(0), -1)
        edge_attr_global_embedding = self.edge_pre(rbf_ohe)

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            batch=batch,
            edge_index=edge_index,
            edge_attr=edge_attr_global_transformed,
            edge_attr_initial_ohe=edge_attr_initial_ohe,
            edge_attr_global_embedding=edge_attr_global_embedding,
            z=z,
            ligand_mask=variable_mask,
            latent_gamma=latent_gamma,
        )

        coords_pred, atoms_pred, bonds_pred, score_pred = self.prediction_head(
            x=out,
            batch=batch,
            edge_index=edge_index,
            edge_index_ligand=edge_index_ligand,
            batch_ligand=batch_ligand,
            ligand_mask=ligand_mask,
            edge_mask_ligand=edge_mask_ligand,
        )

        if self.store_intermediate_coords and self.training:
            pos_list = out["p_list"]
            assert len(pos_list) > 0
            pos_list.append(coords_pred)
            coords_pred = torch.stack(pos_list, dim=0)  # [num_layers, N, 3]

        out = {
            "coords_pred": coords_pred,
            "atoms_pred": atoms_pred,
            "bonds_pred": bonds_pred,
            "score_pred": score_pred,
            "gnn_scalars": out["s"],
        }

        return out


def initialise_model(hparams: dict) -> DenoisingEdgeNetwork:
    if "node_level_t" in hparams.keys():
        node_level_t = hparams["node_level_t"]
    else:
        node_level_t = False
    model = DenoisingEdgeNetwork(
        num_atom_features=hparams["num_atom_types"] + hparams["num_charge_classes"],
        num_bond_types=hparams["num_bond_classes"],
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        edge_dim=hparams["edim"],
        cutoff_local=hparams["cutoff_local"],
        num_layers=hparams["num_layers"],
        latent_dim=hparams["latent_dim"],
        vector_aggr=hparams["vector_aggr"],
        atom_mapping=True,
        bond_mapping=True,
        use_out_norm=hparams["use_out_norm"],
        store_intermediate_coords=hparams["store_intermediate_coords"],
        joint_property_prediction=hparams["joint_property_prediction"],
        regression_property=hparams["regression_property"],
        node_level_t=node_level_t,
    )
    return model


def load_model_from_ckpt(ckpt_path, old: bool = True):
    """Loads a model from the Trainer class.
    Here only the model class is extracted from the e3mol.experiments.diffmodel.Trainer class.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = initialise_model(checkpoint["hyper_parameters"])
    state_dict = checkpoint["state_dict"]
    if old:
        state_dict = {
            re.sub(r"^model\.", "", k): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model")
        }
    else:
        state_dict = {
            re.sub(r"^model\.", "", re.sub(r"^model\.", "", k)): v
            for k, v in state_dict.items()
            if not any(x in k for x in ["prior", "sde", "cat"])
        }
    model.load_state_dict(state_dict)
    return model


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        use_out_norm: bool = False,
    ) -> None:
        super().__init__()

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping:
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        self.sdim, self.vdim = hn_dim

        self.gnn = EQGATLocalGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            edge_dim=edge_dim,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            store_intermediate_coords=False,
            use_out_norm=use_out_norm,
            coords_update=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.gnn.reset_parameters()

    def calculate_edge_attrs(
        self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor, sqrt: bool = True
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos_norm = F.normalize(pos, p=2, dim=-1)
        a = pos_norm[target] * pos_norm[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (d.unsqueeze(-1) + 1.0))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = Tensor,
        batch: OptTensor = None,
    ):
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)

        edge_attr_transformed = self.bond_mapping(edge_attr)
        edge_attr_transformed = self.calculate_edge_attrs(
            edge_index=edge_index,
            edge_attr=edge_attr_transformed,
            pos=pos,
            sqrt=True,
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            edge_index=edge_index,
            edge_attr=edge_attr_transformed,
            batch=batch,
        )
        return out

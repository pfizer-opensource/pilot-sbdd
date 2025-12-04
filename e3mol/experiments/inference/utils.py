from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from biopandas.pdb import PandasPdb
from rdkit import Chem
from torch_geometric.data import Batch, Data

from e3mol.experiments.data.datainfo import ATOM_ENCODER
from e3mol.experiments.utils import prepare_inpainting_ligand_batch
from e3mol.forge.data import MoleculeData, ProteinData
from e3mol.forge.transforms import (
    ProteinFeatureConstants,
    _process_class_pdb,
    mol_structure_to_pyg_dicts,
    protein_structure_to_pyg_dicts,
)


def write_sdf_file(sdf_path, molecules):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)
    w.close()


def subset_pocket_features(prot: Dict) -> Dict:
    assert "pocket_mask" in prot.keys()
    out = {k: v[prot["pocket_mask"]] for k, v in prot.items()}
    return out


def load_ligand_pyg(
    sdf_file: Union[str, Path],
    atom_encoder: Optional[Dict[str, int]] = None,
    keys: Optional[List] = None,
    removeHs_ligand: bool = True,
) -> Data:

    if keys is None:
        keys = [
            "x",
            "pos",
            "charge",
            "hybridization",
            "edge_index",
            "edge_attr",
            "residue_names",
            "element",
            "pocket_mask",
        ]

    if atom_encoder is None:
        atom_encoder = ATOM_ENCODER

    ligand = MoleculeData.from_sdf([sdf_file], removeHs=removeHs_ligand, sanitize=True)
    lig = mol_structure_to_pyg_dicts(ligand, removeHs=removeHs_ligand)[0]
    lig["x"] = torch.tensor([atom_encoder[i] for i in lig["element"]]).long()
    lig["charges"] = lig["charge"]
    return Data.from_dict(lig)


def load_protein_pyg(
    pdb_file: Union[str, Path],
    sdf_file: Union[str, Path],
    select_classes: List[str],
    atom_encoder: Optional[Dict[str, int]] = None,
    keys: Optional[List] = None,
    pocket_cutoff: float = 7.0,
    removeHs_ligand_pre: bool = False,
    removeHs_pocket_pre: bool = False,
    removeHs_ligand_post: bool = True,
    removeHs_pocket_post: bool = True,
    remove_water: bool = True,
) -> Data:

    if keys is None:
        keys = [
            "x",
            "pos",
            "charge",
            "edge_index",
            "edge_attr",
            "residue_names",
            "element",
            "pocket_mask",
        ]
    if atom_encoder is None:
        atom_encoder = ATOM_ENCODER

    prot = ProteinData.from_pdb(
        [pdb_file],
        [sdf_file],
        pocket_cutoff=pocket_cutoff,
        remove_ligand_hydrogens=removeHs_ligand_pre,
        select_classes=select_classes,
        remove_water=remove_water,
    )
    prot = protein_structure_to_pyg_dicts(
        prot,
        select_classes=select_classes,
        removeHs=removeHs_pocket_post,
        remove_water=remove_water,
    )[0]
    prot["x"] = torch.tensor([atom_encoder[i] for i in prot["element"]]).long()
    prot = subset_pocket_features(prot)
    prot = Data.from_dict(
        {key + "_pocket": prot.get(key) for key in keys if key in prot.keys()}
    )
    return prot


def load_protein_ligand_pyg(
    pdb_file: Union[str, Path],
    sdf_file: Union[str, Path],
    select_classes: List[str],
    atom_encoder: Optional[Dict[str, int]] = None,
    keys: Optional[List] = None,
    pocket_cutoff: float = 7.0,
    removeHs_ligand_pre: bool = False,
    removeHs_pocket_pre: bool = False,
    removeHs_ligand_post: bool = True,
    removeHs_pocket_post: bool = True,
    remove_water: bool = True,
) -> Data:

    if keys is None:
        keys = [
            "x",
            "pos",
            "charge",
            "edge_index",
            "edge_attr",
            "residue_names",
            "element",
        ]

    if atom_encoder is None:
        atom_encoder = ATOM_ENCODER

    prot = ProteinData.from_pdb(
        [pdb_file],
        [sdf_file],
        pocket_cutoff=pocket_cutoff,
        remove_ligand_hydrogens=removeHs_ligand_pre,
        select_classes=select_classes,
        remove_water=remove_water,
    )
    ligand = MoleculeData.from_sdf(
        [sdf_file], removeHs=removeHs_ligand_pre, sanitize=True
    )
    lig = mol_structure_to_pyg_dicts(ligand, removeHs=removeHs_ligand_post)[0]
    prot = protein_structure_to_pyg_dicts(
        prot,
        select_classes=select_classes,
        removeHs=removeHs_pocket_post,
        remove_water=remove_water,
    )[0]

    lig["x"] = torch.tensor([atom_encoder[i] for i in lig["element"]]).long()
    lig["charges"] = lig["charge"]
    prot["x"] = torch.tensor([atom_encoder[i] for i in prot["element"]]).long()
    prot = subset_pocket_features(prot)
    lig = Data.from_dict(lig)
    prot = Data.from_dict(
        {key + "_pocket": prot.get(key) for key in keys if key in prot.keys()}
    )
    lig.update(prot)

    return lig


def load_pocket_from_pdb_and_ids(
    pdb_path,
    residue_ids: np.ndarray | list,
    removeHs: bool = True,
    remove_water: bool = True,
    keys: Optional[List] = None,
) -> Data:

    if keys is None:
        keys = [
            "x",
            "pos",
            "charge",
            "edge_index",
            "edge_attr",
            "residue_names",
            "element",
            "pocket_mask",
        ]

    protein = PandasPdb().read_pdb(Path(pdb_path))
    protein_df = protein.df["ATOM"]
    mask = protein_df.residue_number.isin(residue_ids)
    pocket_df = protein_df[mask].reset_index(drop=True)
    if remove_water:
        pocket_df = pocket_df[pocket_df.residue_name != "HOH"].reset_index(drop=True)
    if removeHs:
        pocket_df = pocket_df[pocket_df.element_symbol != "H"].reset_index(drop=True)

    pm = np.ones(len(pocket_df))
    (
        atomic_positions,
        atomic_names,
        residue_names,
        residue_numbers,
        element_names,
        atomic_numbers,
        pm,
    ) = _process_class_pdb(pocket_df, pm)
    additional_features = {
        ProteinFeatureConstants.residue_feat: residue_names,
        ProteinFeatureConstants.residue_number: residue_numbers,
        ProteinFeatureConstants.atomic_names: atomic_names,
        ProteinFeatureConstants.element: element_names,
        ProteinFeatureConstants.pocket_mask: torch.from_numpy(pm).bool(),
    }
    data = {
        ProteinFeatureConstants.atomic_number: torch.from_numpy(atomic_numbers)
        .long()
        .squeeze(),
        ProteinFeatureConstants.atomic_pos: torch.from_numpy(atomic_positions).float(),
        **additional_features,
    }
    prot = {
        k: v.squeeze() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    data = Data.from_dict(
        {key + "_pocket": prot.get(key) for key in keys if key in prot.keys()}
    )
    return data


def create_batch_from_pl_files(
    pdb_file: Union[str, Path],
    sdf_file: Union[str, Path],
    batch_size: int,
    pocket_cutoff: float = 7.0,
    select_classes: List[str] | None = None,
    removeHs_ligand_pre: bool = False,
    removeHs_pocket_pre: bool = False,
    removeHs_ligand_post: bool = True,
    removeHs_pocket_post: bool = True,
    remove_water: bool = True,
    inpainting: bool = False,
    keep_ids: Optional[List[int]] = None,
    min_nodes_bias: int = 0,
    max_nodes_bias: int = 0,
    inpaint_file: Union[str, Path, None] = None,
    anchor_idx: Optional[List[int]] = None,
    atom_encoder: Optional[Dict[str, int]] = None,
    keys: Optional[List] = None,
) -> Union[Data, Batch]:
    """_summary_

    Args:
        pdb_file (Union[str, Path]): _description_
        sdf_file (Union[str, Path]): _description_
        batch_size (int): _description_
        pocket_cutoff (float, optional): _description_. Defaults to 7.0.
        select_classes (List[str] | None, optional): _description_. Defaults to None.
        removeHs_ligand_pre (bool, optional): _description_. Defaults to False.
        removeHs_pocket_pre (bool, optional): _description_. Defaults to False.
        removeHs_ligand_post (bool, optional): _description_. Defaults to True.
        removeHs_pocket_post (bool, optional): _description_. Defaults to True.
        remove_water (bool, optional): _description_. Defaults to True.
        inpainting (bool, optional): _description_. Defaults to False.
        keep_ids (Optional[List[int]], optional): _description_. Defaults to None.
        min_nodes_bias (int, optional): _description_. Defaults to 0.
        max_nodes_bias (int, optional): _description_. Defaults to 0.
        inpaint_file (Union[str, Path, None], optional): _description_. Defaults to None.
        anchor_idx (Optional[List[int]], optional): _description_. Defaults to None.
        atom_encoder (Optional[Dict[str, int]], optional): _description_. Defaults to None.
        keys (Optional[List], optional): _description_. Defaults to None.

    Returns:
        Union[Data, Batch]: _description_
    """

    if select_classes is None:
        select_classes = ["ATOM"]

    if inpainting:
        assert keep_ids is not None, "keep_ids must be provided for inpainting"
        data = load_ligand_pyg(sdf_file, atom_encoder, keys, removeHs_ligand_post)
        protein_data = load_protein_pyg(
            pdb_file,
            sdf_file,
            select_classes,
            atom_encoder,
            keys,
            pocket_cutoff,
            removeHs_ligand_pre,
            removeHs_pocket_pre,
            removeHs_ligand_post,
            removeHs_pocket_post,
            remove_water,
        )
        if inpaint_file is not None:
            data = load_ligand_pyg(
                inpaint_file, atom_encoder, keys, removeHs_ligand_post
            )
        data = data.update(protein_data)

        batch = prepare_inpainting_ligand_batch(
            data=data,
            min_nodes_bias=min_nodes_bias,
            max_nodes_bias=max_nodes_bias,
            num_graphs=batch_size,
            keep_ids=keep_ids,
            anchor_idx=anchor_idx,
        )
    else:
        data = load_protein_ligand_pyg(
            pdb_file,
            sdf_file,
            select_classes,
            atom_encoder,
            keys,
            pocket_cutoff,
            removeHs_ligand_pre,
            removeHs_pocket_pre,
            removeHs_ligand_post,
            removeHs_pocket_post,
            remove_water,
        )
        batch = Batch.from_data_list(
            [deepcopy(data) for _ in range(batch_size)],
            follow_batch=["pos", "pos_pocket"],
        )
    return batch

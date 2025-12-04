from typing import Dict, List, Optional, Union

import numpy as np
import torch
from biopandas.pdb import PandasPdb
from rdkit import Chem
from torch import Tensor

from e3mol.experiments.data.datainfo import ADDITIONAL_FEATS_MAP as X_MAP
from e3mol.forge.data import MoleculeData, ProteinData
from e3mol.forge.utils import convert_atomic_element_to_number


class MoleculeFeatureConstants:
    element = "element"
    charge = "charge"
    edge_index = "edge_index"
    edge_attr = "edge_attr"
    atomic_number = "x"
    atomic_pos = "pos"
    hybridization = "hybridization"


def rdmol_to_pyg_dict(
    mol: Chem.Mol, removeHs: bool = True, has_conformer: bool = True
) -> Dict[str, Union[Optional[Tensor], List]]:

    if removeHs:
        mol = Chem.RemoveHs(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)

    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)).float()
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()
    if has_conformer:
        atomic_positions = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    else:
        atomic_positions = None

    atomic_numbers = []
    atomic_elements = []
    atomic_charges = []
    atomic_hybridizations = []

    for atom in mol.GetAtoms():
        atomic_numbers.append(atom.GetAtomicNum())
        atomic_elements.append(atom.GetSymbol())
        atomic_charges.append(atom.GetFormalCharge())
        atomic_hybridizations.append(
            X_MAP["hybridization"].index(atom.GetHybridization())
        )

    atomic_numbers = torch.tensor(atomic_numbers).long()
    atomic_charges = torch.tensor(atomic_charges).long()
    atomic_hybridizations = torch.tensor(atomic_hybridizations).long()

    data = {
        MoleculeFeatureConstants.atomic_number: atomic_numbers,
        MoleculeFeatureConstants.atomic_pos: atomic_positions,
        MoleculeFeatureConstants.charge: atomic_charges,
        MoleculeFeatureConstants.element: atomic_elements,
        MoleculeFeatureConstants.hybridization: atomic_hybridizations,
        MoleculeFeatureConstants.edge_index: edge_index,
        MoleculeFeatureConstants.edge_attr: edge_attr,
    }
    data = {
        k: v.squeeze() if isinstance(v, Tensor) and "edge" not in k else v
        for k, v in data.items()
    }

    return data


def mol_structure_to_pyg_dicts(
    molecule: MoleculeData, removeHs: bool = True, has_conformer: bool = True
) -> List[Dict[str, Union[Optional[Tensor], List]]]:
    pyg_dicts = []
    for mol in molecule.rdkit_mols:
        pyg_dicts.append(rdmol_to_pyg_dict(mol, removeHs, has_conformer))
    return pyg_dicts


class ProteinFeatureConstants:
    residue_feat = "residue_names"
    residue_number = "residue_number"
    atomic_names = "atomic_names"
    element = "element"
    atomic_number = "x"
    atomic_pos = "pos"
    pocket_mask = "pocket_mask"


def _process_class_pdb(df, pm, remove_water=True, removeHs=True):
    if remove_water:
        water_mask = ~(df.residue_name == "HOH").values
    else:
        water_mask = np.ones(len(df), dtype=bool)

    if removeHs:
        heavy_elements_mask = ~(df.element_symbol == "H").values
    else:
        heavy_elements_mask = np.ones(len(df), dtype=bool)

    pm = pm * (heavy_elements_mask * water_mask)

    atomic_positions = df[["x_coord", "y_coord", "z_coord"]].values
    atomic_names = df[["atom_name"]].values.flatten()
    residue_names = df[["residue_name"]].values.flatten()
    residue_numbers = df[["residue_number"]].values.flatten()
    element_names = df[["element_symbol"]].values.flatten()
    atomic_numbers = np.array(
        [convert_atomic_element_to_number(e) for e in element_names]
    )

    return (
        atomic_positions,
        atomic_names,
        residue_names,
        residue_numbers,
        element_names,
        atomic_numbers,
        pm,
    )


def ppdb_to_pyg_dict(
    ppdb: PandasPdb,
    select_classes: List[str],
    pocket_mask: Dict[str, np.ndarray],
    removeHs: bool = True,
    remove_water: bool = True,
) -> Dict[str, Optional[Tensor]]:

    residues_names_l = []
    residues_numbers_l = []
    atomic_names_l = []
    element_names_l = []
    atomic_numbers_l = []
    atomic_positions_l = []
    pocket_masks_l = []

    for select_class in select_classes:
        df = ppdb.df[select_class].copy()
        pm = pocket_mask[select_class]
        (
            atomic_positions,
            atomic_names,
            residue_names,
            residue_numbers,
            element_names,
            atomic_numbers,
            pm,
        ) = _process_class_pdb(df, pm, remove_water, removeHs)
        residues_names_l.append(residue_names)
        residues_numbers_l.append(residue_numbers)
        atomic_names_l.append(atomic_names)
        element_names_l.append(element_names)
        atomic_numbers_l.append(atomic_numbers)
        atomic_positions_l.append(atomic_positions)
        pocket_masks_l.append(pm)

    residue_names = np.concatenate(residues_names_l)
    residue_numbers = np.concatenate(residues_numbers_l)
    atomic_names = np.concatenate(atomic_names_l)
    element_names = np.concatenate(element_names_l)
    atomic_numbers = np.concatenate(atomic_numbers_l)
    atomic_positions = np.concatenate(atomic_positions_l)
    pocket_mask = np.concatenate(pocket_masks_l)

    additional_features = {
        ProteinFeatureConstants.residue_feat: residue_names,
        ProteinFeatureConstants.residue_number: residue_numbers,
        ProteinFeatureConstants.atomic_names: atomic_names,
        ProteinFeatureConstants.element: element_names,
        ProteinFeatureConstants.pocket_mask: torch.from_numpy(pocket_mask).bool(),
    }
    data = {
        ProteinFeatureConstants.atomic_number: torch.from_numpy(atomic_numbers)
        .long()
        .squeeze(),
        ProteinFeatureConstants.atomic_pos: torch.from_numpy(atomic_positions).float(),
        **additional_features,
    }
    data = {k: v.squeeze() if isinstance(v, Tensor) else v for k, v in data.items()}
    return data


def protein_structure_to_pyg_dicts(
    protein: ProteinData,
    select_classes: List[str],
    removeHs: bool = True,
    remove_water: bool = True,
) -> List[Dict[str, Optional[Tensor]]]:
    pyg_dicts = []
    for _protein, _pocket_mask in zip(protein.pdbs, protein._pocket_masks):
        pyg_dicts.append(
            ppdb_to_pyg_dict(
                _protein, select_classes, _pocket_mask, removeHs, remove_water
            )
        )
    return pyg_dicts

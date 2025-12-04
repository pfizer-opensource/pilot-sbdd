import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
from torch_geometric.data import Batch, Data

from e3mol.experiments.data.datainfo import ADDITIONAL_FEATS_MAP as x_map
from e3mol.experiments.data.datainfo import ATOM_ENCODER
from e3mol.forge.data import MoleculeData, ProteinData
from e3mol.forge.transforms import (
    mol_structure_to_pyg_dicts,
    protein_structure_to_pyg_dicts,
)

fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

RDLogger.DisableLog("rdApp.*")


def subset_pocket_features(prot: Dict) -> Dict:
    assert "pocket_mask" in prot.keys()
    out = {k: v[prot["pocket_mask"]] for k, v in prot.items()}
    return out


def load_protein_ligand_pyg(
    pdb_file: Union[str, Path],
    sdf_file: Union[str, Path],
    atom_encoder: Optional[Dict[str, int]] = None,
    keys: Optional[List] = None,
    pocket_cutoff: float = 7.0,
    removeHs_ligand: bool = True,
    removeHs_pocket: bool = True,
) -> Data:

    if keys is None:
        keys = ["x", "pos", "edge_index", "edge_attr", "residue_names", "element"]

    if atom_encoder is None:
        atom_encoder = ATOM_ENCODER

    prot = ProteinData.from_pdb([pdb_file], [sdf_file], pocket_cutoff=pocket_cutoff)
    ligand = MoleculeData.from_sdf([sdf_file])
    lig = mol_structure_to_pyg_dicts(ligand, removeHs=removeHs_ligand)[0]
    prot = protein_structure_to_pyg_dicts(prot, removeHs=removeHs_pocket)[0]

    lig["x"] = torch.tensor([atom_encoder[i] for i in lig["element"]]).long()
    prot["x"] = torch.tensor([atom_encoder[i] for i in prot["element"]]).long()
    prot = subset_pocket_features(prot)

    lig = Data.from_dict(
        {key + "_ligand": lig.get(key) for key in keys if key in lig.keys()}
    )
    # required since PyG needs the attribute in case we want to use Batch.from_data_list()
    lig.x = lig.x_ligand
    prot = Data.from_dict(
        {key + "_pocket": prot.get(key) for key in keys if key in prot.keys()}
    )
    lig.update(prot)

    return lig


def create_pl_batch(pl_data: Data, batch_size: int) -> Batch:
    data_list = [pl_data for _ in range(batch_size)]
    return Batch.from_data_list(data_list, follow_batch=["x_ligand", "x_pocket"])


def mol_to_torch_geometric(
    mol,
    atom_encoder,
    smiles=None,
    remove_hydrogens: bool = True,
    cog_proj: bool = False,
    add_ad=True,
    kekulize: bool = True,
    **kwargs,
):

    # assert remove_hydrogens, "remove_hydrogens=True should be used for all experiments"

    if remove_hydrogens:
        mol = Chem.RemoveAllHs(mol)
        if kekulize:
            Chem.Kekulize(mol, clearAromaticFlags=True)
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    if remove_hydrogens and kekulize:
        assert max(bond_types) != 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    if cog_proj:
        pos = pos - torch.mean(pos, dim=0, keepdim=True)

    atom_types = []
    all_charges = []
    is_aromatic = []
    is_in_ring = []
    sp_hybridization = []
    num_attached_Hs = []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())
        is_aromatic.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        is_in_ring.append(x_map["is_in_ring"].index(atom.IsInRing()))
        sp_hybridization.append(x_map["hybridization"].index(atom.GetHybridization()))
        # retrieve the total number of attached Hs which can be either explicit or implicit
        num_attached_Hs.append(x_map["numHs"].index(atom.GetTotalNumHs()))

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    is_aromatic = torch.Tensor(is_aromatic).long()
    is_in_ring = torch.Tensor(is_in_ring).long()
    hybridization = torch.Tensor(sp_hybridization).long()
    numHs = torch.Tensor(num_attached_Hs).long()

    if add_ad:
        # hydrogen bond acceptor and donor
        feats = factory.GetFeaturesForMol(mol)
        donor_ids = []
        acceptor_ids = []
        for f in feats:
            if f.GetFamily().lower() == "donor":
                donor_ids.append(f.GetAtomIds())
            elif f.GetFamily().lower() == "acceptor":
                acceptor_ids.append(f.GetAtomIds())

        if len(donor_ids) > 0:
            donor_ids = np.concatenate(donor_ids)
        else:
            donor_ids = np.array([])

        if len(acceptor_ids) > 0:
            acceptor_ids = np.concatenate(acceptor_ids)
        else:
            acceptor_ids = np.array([])
        is_acceptor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
        is_donor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
        if len(donor_ids) > 0:
            is_donor[donor_ids] = 1
        if len(acceptor_ids) > 0:
            is_acceptor[acceptor_ids] = 1

        is_donor = torch.from_numpy(is_donor).long()
        is_acceptor = torch.from_numpy(is_acceptor).long()
    else:
        is_donor = is_acceptor = None

    additional: Dict[str, torch.Tensor] = {}

    data = Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        charges=all_charges,
        smiles=smiles,
        is_aromatic=is_aromatic,
        is_in_ring=is_in_ring,
        is_h_donor=is_donor,
        is_h_acceptor=is_acceptor,
        hybridization=hybridization,
        numHs=numHs,
        mol=mol,
        **additional,
    )

    return data

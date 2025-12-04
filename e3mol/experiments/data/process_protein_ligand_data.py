from typing import Dict, Union

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
from rdkit import Chem

from e3mol.experiments.data.datainfo import AA_ENCODER as amino_acid_dict


def process_protein_ligand(
    pdb_file,
    sdf_file,
    dist_cutoff: float = 5.0,
    ca_only: bool = False,
    no_H: bool = True,
    residue_com: bool = False,
):

    pdb_struct = PDBParser(QUIET=True).get_structure("", pdb_file)

    ligand = Chem.SDMolSupplier(str(sdf_file), removeHs=no_H, sanitize=True)[0]
    if not no_H:
        ligand = Chem.AddHs(ligand, addCoords=True)
    if ligand is None:
        print(
            f"SDF file containts invalid ligand that cannot be sanitized ({sdf_file})"
        )

    lig_atoms = np.array([a.GetSymbol() for a in ligand.GetAtoms()])
    lig_coords = np.array(
        [
            list(ligand.GetConformer(0).GetAtomPosition(idx))
            for idx in range(ligand.GetNumAtoms())
        ]
    )

    ligand_data: Dict[str, Union[np.ndarray, Chem.Mol]] = {
        "lig_coords": lig_coords,
        "lig_atoms": lig_atoms,
        "lig_mol": ligand,
    }

    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])

        if residue_com:
            res_coords = residue.center_of_mass()
            res_coords = res_coords[np.newaxis, :]

        if (
            is_aa(residue.get_resname(), standard=True)
            and (
                ((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5
            ).min()
            < dist_cutoff
        ):
            pocket_residues.append(residue)

    pocket_chainids = [f"{res.id[1]}.{res.parent.id}" for res in pocket_residues]
    pocket_resnames = [res.get_resname() for res in pocket_residues]
    pocket_resid = [res.id[1] for res in pocket_residues]

    if ca_only:
        try:
            pocket_one_hot = []
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == "CA":
                        pocket_one_hot.append(
                            np.eye(
                                1,
                                len(amino_acid_dict),
                                amino_acid_dict[three_to_one.get(res.get_resname())],
                            ).squeeze()
                        )
                        full_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            full_coords = np.stack(full_coords)
        except KeyError as e:
            raise KeyError(f"{e} not in amino acid dict ({pdb_file}, {sdf_file})")
        pocket_data = {
            "pocket_coords": full_coords,
            "pocket_one_hot": pocket_one_hot,
            "pocket_chainids": pocket_chainids,
        }
    else:
        # c-alphas and residue idendity
        pocket_one_hot = []
        ca_mask = []
        # full
        full_atoms = []
        full_atom_names = []
        full_coords = []
        m = False
        for res in pocket_residues:
            for atom in res.get_atoms():
                if atom.name == "CA":
                    pocket_one_hot.append(
                        np.eye(
                            1,
                            len(amino_acid_dict),
                            amino_acid_dict[three_to_one.get(res.get_resname())],
                        ).squeeze()
                    )
                    m = True
                else:
                    m = False
                ca_mask.append(m)
                full_atoms.append(atom.element)
                full_atom_names.append(atom.name)
                full_coords.append(atom.coord)
        pocket_one_hot = np.stack(pocket_one_hot, axis=0)
        full_atoms = np.stack(full_atoms, axis=0)
        full_atom_names = np.stack(full_atom_names, axis=0)
        full_coords = np.stack(full_coords, axis=0)
        ca_mask = np.array(ca_mask, dtype=bool)
        if no_H:
            indices_H = np.where(full_atoms == "H")
            if indices_H[0].size > 0:
                mask = np.ones(full_atoms.size, dtype=bool)  # type: ignore
                mask[indices_H] = False
                full_atoms = full_atoms[mask]
                full_atom_names = full_atom_names[mask]
                full_coords = full_coords[mask]
                ca_mask = ca_mask[mask]
        assert sum(ca_mask) == pocket_one_hot.shape[0]  # type: ignore
        assert len(full_atoms) == len(full_coords)
        pocket_data = {
            "pocket_coords": full_coords,
            "pocket_resnames": pocket_resnames,
            "pocket_chainids": pocket_chainids,
            "pocket_resids": pocket_resid,
            "pocket_atoms": full_atoms,
            "pocket_atom_names": full_atom_names,
            "pocket_one_hot": pocket_one_hot,
            "pocket_ca_mask": ca_mask,
        }

        return ligand_data, pocket_data


def save_all_as_np(
    file_name,
    pdb_and_mol_ids,
    lig_coords,
    lig_atom,
    lig_mask,
    lig_mol,
    pocket_coords,
    pocket_atom,
    pocket_atom_names,
    pocket_mask,
    pocket_resids,
    pocket_chainids,
    pocket_resnames,
    pocket_one_hot,
    pocket_ca_mask,
    **kwargs,
):
    np.savez(
        file_name,
        names=pdb_and_mol_ids,
        lig_coords=lig_coords,
        lig_atom=lig_atom,
        lig_mask=lig_mask,
        lig_mol=lig_mol,
        pocket_coords=pocket_coords,
        pocket_atom=pocket_atom,
        pocket_atom_names=pocket_atom_names,
        pocket_mask=pocket_mask,
        pocket_resids=pocket_resids,
        pocket_chainids=pocket_chainids,
        pocket_resnames=pocket_resnames,
        pocket_one_hot=pocket_one_hot,
        pocket_ca_mask=pocket_ca_mask,
        **kwargs,
    )
    return True

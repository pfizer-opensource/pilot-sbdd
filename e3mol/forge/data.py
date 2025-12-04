import os
import shutil
import tempfile
from itertools import chain
from pathlib import Path
from typing import Any, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors
from rdkit.Chem.rdmolfiles import SDWriter

from e3mol.forge.core import Data

PathLike = Union[Path, str]
D = TypeVar("D", bound="Data")

__all__ = ["MoleculeData", "ProteinData"]


def concat(values: List[np.ndarray | List[Any]]) -> np.ndarray | List[Any]:
    """
    Concatenate a list of properties into a single array.
    """
    if isinstance(values[0], np.ndarray):
        return np.concatenate(values, axis=0)
    return list(chain(*values))


class MoleculeData(Data["MoleculeData"]):
    """
    Data object describing molecules.

    """

    def __init__(self, rdkit_mols: List[Chem.Mol]) -> None:
        """
        Args:
        - mols: A list of RDKit molecules.
        """
        super().__init__()
        self.rdkit_mols = rdkit_mols

    def __len__(self) -> int:
        return len(self.rdkit_mols)

    @property
    def ids(self) -> List[str]:
        return ["molecule"]

    def __getitem__(self, idx: List[int] | int | np.ndarray) -> "MoleculeData":
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        mols = [self.rdkit_mols[i] for i in idx]

        data = MoleculeData(mols)
        data.properties = {k: [v[i] for i in idx] for k, v in self.properties.items()}
        return data

    def __setitem__(
        self, idx: List[int] | int | np.ndarray, value: "MoleculeData"
    ) -> None:
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        if len(idx) != len(value):
            raise ValueError("Length of index must match length of supplied data.")
        if max(idx) >= len(self):
            raise ValueError("Index out of bounds.")

        # set molecules
        for i, j in zip(idx, range(len(value))):
            self.rdkit_mols[i] = value.rdkit_mols[j]
        # set properties
        for k, v in value.properties.items():
            for i in idx:
                self.properties[k][i] = v[i]

    @staticmethod
    def concatenate(data: List["MoleculeData"]) -> "MoleculeData":
        """
        Concatenate multiple MoleculeData objects.
        """
        mols = []
        for d in data:
            mols += d.rdkit_mols

        new_data = MoleculeData(mols)
        new_data.properties = {
            k: concat([d.properties[k] for d in data])
            for k in data[0].properties.keys()
        }
        return new_data

    def repeat(self, n: int) -> "MoleculeData":
        """
        Repeat the molecule data n times.
        """
        mols = self.rdkit_mols * n
        return MoleculeData(mols)

    def repeat_interleave(self, n: int) -> "MoleculeData":
        """
        Repeat the molecule data n times interleaved.
        """
        mols = np.repeat(self.rdkit_mols, n).tolist()
        return MoleculeData(mols)

    def generate_conformers(
        self,
        num_conformers=10,
        maxAttempts=1000,
        pruneRmsThresh=0.1,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        enforceChirality=True,
        numThreads=0,
    ):
        new_mols = []
        for mol in self.rdkit_mols:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=num_conformers,
                maxAttempts=maxAttempts,
                pruneRmsThresh=pruneRmsThresh,
                useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
                useBasicKnowledge=useBasicKnowledge,
                enforceChirality=enforceChirality,
                numThreads=numThreads,
            )
            new_mols.append(mol)
        self.rdkit_mols = new_mols

    def to_arrow(self) -> pa.Table:
        arr = pa.array([Chem.MolToJSON(m) for m in self.rdkit_mols], type=pa.string())
        table = pa.Table.from_arrays(
            [arr],
            ["molecule"],
            metadata={"_decoder": self.__module__ + "." + self.__class__.__name__},
        )
        table = self._add_properties_to_table(table)

        return table

    @staticmethod
    def from_arrow(table: pa.Table) -> "MoleculeData":
        data = MoleculeData(
            [
                Chem.JSONToMols(mol_string)[0]
                for mol_string in table["molecule"].to_pylist()
            ]
        )
        data._load_properties_from_table(table)
        return data

    @staticmethod
    def from_sdf(
        sdf_files: List[PathLike], removeHs=False, sanitize=True
    ) -> "MoleculeData":
        mols = []
        for sdf in sdf_files:
            mols += [
                mol
                for mol in Chem.SDMolSupplier(
                    str(sdf), removeHs=removeHs, sanitize=sanitize
                )
            ]
        return MoleculeData(mols)

    # chemical properties from RDKit
    @property
    def num_heavy_atoms(self) -> np.ndarray:
        """Number of heavy atoms in the molecule."""
        return np.array([mol.GetNumHeavyAtoms() for mol in self.rdkit_mols])

    @property
    def num_atoms(self) -> np.ndarray:
        """Number of atoms in the molecule."""
        return np.array([mol.GetNumAtoms() for mol in self.rdkit_mols])

    @property
    def num_bonds(self) -> np.ndarray:
        """Number of bonds in the molecule."""
        return np.array([mol.GetNumBonds() for mol in self.rdkit_mols])

    @property
    def num_rings(self) -> np.ndarray:
        """Number of rings in the molecule."""
        return np.array([mol.GetRingInfo().NumRings() for mol in self.rdkit_mols])

    @property
    def num_rotatable_bonds(self) -> np.ndarray:
        """Number of rotatable bonds in the molecule."""
        return np.array(
            [rdMolDescriptors.CalcNumRotatableBonds(mol) for mol in self.rdkit_mols]
        )

    @property
    def num_h_donors(self) -> np.ndarray:
        """Number of hydrogen bond donors in the molecule."""
        return np.array([rdMolDescriptors.CalcNumHBD(mol) for mol in self.rdkit_mols])

    @property
    def num_h_acceptors(self) -> np.ndarray:
        """Number of hydrogen bond acceptors in the molecule."""
        return np.array([rdMolDescriptors.CalcNumHBA(mol) for mol in self.rdkit_mols])

    @property
    def num_aromatic_rings(self) -> np.ndarray:
        """Number of aromatic rings in the molecule."""
        return np.array(
            [rdMolDescriptors.CalcNumAromaticRings(mol) for mol in self.rdkit_mols]
        )

    @property
    def num_aliphatic_rings(self) -> np.ndarray:
        """Number of aliphatic rings in the molecule."""
        return np.array(
            [rdMolDescriptors.CalcNumAliphaticRings(mol) for mol in self.rdkit_mols]
        )

    @property
    def num_atom_stereocenters(self) -> np.ndarray:
        """Number of atom stereocenters in the molecule."""
        return np.array(
            [
                len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                for mol in self.rdkit_mols
            ]
        )

    @property
    def csp3_fraction(self) -> np.ndarray:
        """Fraction of sp3-hybridized carbons in the molecule."""
        return np.array(
            [rdMolDescriptors.CalcFractionCSP3(mol) for mol in self.rdkit_mols]
        )

    @property
    def slogp(self) -> np.ndarray:
        """Calculated logP of the molecule."""
        return np.array([Crippen.MolLogP(mol) for mol in self.rdkit_mols])

    @property
    def num_heteroatoms(self) -> np.ndarray:
        """Number of heteroatoms in the molecule."""
        return np.array(
            [rdMolDescriptors.CalcNumHeteroatoms(mol) for mol in self.rdkit_mols]
        )

    @property
    def molecular_weight(self) -> np.ndarray:
        """Molecular weight of the molecule."""
        return np.array(
            [rdMolDescriptors.CalcExactMolWt(mol) for mol in self.rdkit_mols]
        )

    @property
    def tpsa(self) -> np.ndarray:
        """Topological polar surface area of the molecule."""
        return np.array([rdMolDescriptors.CalcTPSA(mol) for mol in self.rdkit_mols])

    def to_complex_pdb(
        self, dstdir: str, protein: PandasPdb, prefix: str = "complex"
    ) -> List[str]:
        import pymol

        tmpdir = tempfile.mkdtemp(dir="/dev/shm")
        protein_file = os.path.join(tmpdir, "protein.pdb")
        protein.to_pdb(protein_file)

        files = []
        for i in range(len(self)):
            ligand_file = os.path.join(tmpdir, f"ligand_{i}.sdf")
            complex_file = os.path.join(dstdir, f"{prefix}_{i}.pdb")
            files.append(complex_file)

            writer = SDWriter(ligand_file)
            writer.write(self.rdkit_mols[i])
            writer.close()

            pymol.cmd.reinitialize()
            pymol.cmd.load(protein_file, "prot")
            pymol.cmd.load(ligand_file, "ligand")
            pymol.cmd.create("complex", "ligand, prot")
            pymol.cmd.save(complex_file, "complex")
        shutil.rmtree(tmpdir)
        return files

    def to_sdf(self, outpath: str, name: str = "ligand") -> str:
        writer = Chem.SDWriter(outpath)
        for i, mol in enumerate(self.rdkit_mols):
            if not mol.HasProp("_Name"):
                mol.SetProp("_Name", name + "-" + str(i))
            for k, v in self.properties.items():
                mol.SetProp(k, str(v[i]))
            writer.write(mol)
        writer.close()
        return outpath


class ProteinData(Data["ProteinData"]):
    """
    Data object describing protein structures with optional pocket masks.
    """

    def __init__(
        self, pdbs: List[PandasPdb], pocket_masks: Optional[List[np.ndarray]] = None
    ) -> None:
        super().__init__()
        self.pdbs = pdbs
        if pocket_masks is not None:
            if len(self.pdbs) != len(pocket_masks):
                raise ValueError(
                    "Number of pocket masks must match number of proteins."
                )
            self._pocket_masks = [
                np.array([]) if mask is None else mask for mask in pocket_masks
            ]
        else:
            self._pocket_masks = [np.array([]) for _ in range(len(self.pdbs))]

    def __len__(self) -> int:
        return len(self.pdbs)

    @property
    def ids(self) -> List[str]:
        return ["protein", "pocket_mask"]

    def __getitem__(self, idx: List[int] | int | np.ndarray) -> "ProteinData":
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        proteins = [self.pdbs[i] for i in idx]
        pocket_masks = [self._pocket_masks[i] for i in idx]
        data = ProteinData(proteins, pocket_masks)
        data.properties = {k: [v[i] for i in idx] for k, v in self.properties.items()}
        return data

    def __setitem__(
        self, idx: List[int] | int | np.ndarray, value: "ProteinData"
    ) -> None:
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        if len(idx) != len(value):
            raise ValueError("Length of index must match length of supplied data.")
        if max(idx) >= len(self):
            raise ValueError("Index out of bounds.")

        # set proteins
        for i, j in zip(idx, range(len(value))):
            self.pdbs[i] = value.pdbs[j]
            self._pocket_masks[i] = value._pocket_masks[j]
        # set properties
        for k, v in value.properties.items():
            for i in idx:
                self.properties[k][i] = v[i]

    @staticmethod
    def concatenate(data: List["ProteinData"]) -> "ProteinData":
        """
        Concatenate multiple ProteinData objects.
        """
        pdbs = []
        pocket_masks = []
        for d in data:
            pdbs += d.pdbs
            pocket_masks += d._pocket_masks

        new_data = ProteinData(pdbs, pocket_masks)
        new_data.properties = {
            k: concat([d.properties[k] for d in data])
            for k in data[0].properties.keys()
        }
        return new_data

    def repeat(self, n: int) -> "ProteinData":
        """
        Repeat the protein data n times.
        """
        pdbs = self.pdbs * n
        pocket_masks = self._pocket_masks * n
        return ProteinData(pdbs, pocket_masks)

    def repeat_interleave(self, n: int) -> "ProteinData":
        """
        Repeat the protein data n times interleaved.
        """
        pdbs = np.repeat(self.pdbs, n).tolist()
        pocket_masks = np.repeat(self._pocket_masks, n).tolist()
        return ProteinData(pdbs, pocket_masks)

    def get_centers(self, idx: np.ndarray | None = None) -> np.ndarray:
        """
        Returns the centers of the protein pockets.
        """
        idx = idx if idx is not None else np.arange(len(self))
        centers = []
        for i in idx:
            centers.append(self.get_center(i))
        return np.array(centers)

    def get_center(self, idx: int) -> np.ndarray:
        """
        Returns the center of the specified protein pocket.
        """
        protein = self.pdbs[idx]
        mask = self._pocket_masks[idx]

        if mask.shape[0] == 0:
            raise ValueError("Pocket mask must be set")
        center = protein.df["ATOM"][["x_coord", "y_coord", "z_coord"]].values
        center = center[mask].mean(axis=0)
        return center

    def append(
        self, proteins: List[PandasPdb], pocket_masks: Optional[List[np.ndarray]] = None
    ):
        """
        Append proteins to the data object.
        """
        self.pdbs += proteins
        pocket_masks = pocket_masks or [np.array([]) for _ in range(len(proteins))]
        self._pocket_masks += pocket_masks

    def to_arrow(self) -> pa.Table:
        pdbs = []
        for protein in self.pdbs:
            pdbs.append(protein.pdb_text)

        pdbs = pa.array(pdbs, type=pa.string())
        mask_list = [
            mask.tolist() if mask is not None else pa.list_
            for mask in self._pocket_masks
        ]
        masks = pa.array(
            mask_list,
            type=pa.list_(pa.bool_()),
        )

        table = pa.Table.from_arrays(
            [pdbs, masks],
            ["protein", "pocket_mask"],
            metadata={"_decoder": self.__module__ + "." + self.__class__.__name__},
        )
        table = self._add_properties_to_table(table)
        return table

    @staticmethod
    def from_arrow(table: pa.Table) -> "ProteinData":
        proteins = [
            PandasPdb().read_pdb_from_list(str(pdb).splitlines(keepends=True))
            for pdb in table["protein"]
        ]
        masks = [np.array(mask) for mask in table["pocket_mask"].to_pylist()]
        data = ProteinData(proteins, masks)
        data._load_properties_from_table(table)
        return data

    def apply_pocket_masks(
        self,
        mols: List[Chem.Mol],
        pocket_cutoff: float,
        select_classes: List[str],
        remove_ligand_hydrogens: bool = False,
        remove_water: bool = True,
    ):
        """
        Extract pocket masks from the protein structures based on the reference
        ligand positions.

        Args:
        - mols: A list of RDKit molecules representing the ligands.
        - pocket_cutoff: The distance cutoff to define the pocket.
        - remove_ligand_hydrogens: Whether to remove hydrogens from the ligands.
        """
        if len(mols) != len(self.pdbs):
            raise ValueError("Number of ligands must match number of proteins.")

        for i in range(len(self.pdbs)):
            mol = mols[i]
            if remove_ligand_hydrogens:
                mol = Chem.RemoveHs(mol)

            self.apply_pocket_mask(
                i,
                select_classes,
                mol.GetConformer().GetPositions(),
                pocket_cutoff,
                remove_water,
            )

    def apply_pocket_mask(
        self,
        idx: int,
        select_classes: List[str],
        lig_pos: np.ndarray,
        pocket_cutoff: float,
        remove_water: bool = True,
    ):
        """
        Extract pocket mask from the protein structure based on the reference
        ligand positions.

        Args:
        - idx: The index of the protein structure.
        - lig_pos: The positions of the ligand atoms.
        - pocket_cutoff: The distance cutoff to define the pocket.
        """

        outs = {}
        for select_class in select_classes:
            pocket_mask = self.apply_pocket_mask_per_type(
                idx, select_class, lig_pos, pocket_cutoff, remove_water
            )
            outs[select_class] = pocket_mask
        self._pocket_masks[idx] = outs

    def apply_pocket_mask_per_type(
        self,
        idx: int,
        select_class: str,
        lig_pos: np.ndarray,
        pocket_cutoff: float,
        remove_water: bool = True,
    ):
        """
        Extract pocket mask from the protein structure based on the reference
        ligand positions.

        Args:
        - idx: The index of the protein structure.
        - lig_pos: The positions of the ligand atoms.
        - pocket_cutoff: The distance cutoff to define the pocket.
        """

        residue_dfs = []
        prot_df = self.pdbs[idx].df[select_class]
        for _, residue in prot_df.groupby(["chain_id", "residue_number"]):
            if remove_water:
                if residue.residue_name.values[0] == "HOH":
                    residue["is_pocket"] = False
                else:
                    d = np.sqrt(
                        np.sum(
                            (
                                residue[["x_coord", "y_coord", "z_coord"]].values[
                                    :, np.newaxis, :
                                ]
                                - lig_pos[np.newaxis, :, :]
                            )
                            ** 2,
                            axis=-1,
                        )
                    )
                    # if any atom in the residue is within the cutoff distance,
                    # the residue (with all its atoms) is considered to belong to the pocket
                    residue["is_pocket"] = bool(d.min() < pocket_cutoff)
            else:
                d = np.sqrt(
                    np.sum(
                        (
                            residue[["x_coord", "y_coord", "z_coord"]].values[
                                :, np.newaxis, :
                            ]
                            - lig_pos[np.newaxis, :, :]
                        )
                        ** 2,
                        axis=-1,
                    )
                )
                # if any atom in the residue is within the cutoff distance,
                # the residue (with all its atoms) is considered to belong to the pocket
                residue["is_pocket"] = bool(d.min() < pocket_cutoff)
            residue_dfs.append(residue)

        prot_df_processed = pd.concat(residue_dfs, axis=0)
        # join to match chain_id, residue_number and line_idx
        prot_df_processed = pd.merge(
            left=prot_df,
            right=prot_df_processed,
            left_on=prot_df.columns.values.tolist(),
            right_on=prot_df.columns.values.tolist(),
        )
        self.pdbs[idx].df[select_class] = prot_df_processed
        pocket_mask = prot_df_processed.is_pocket.values.flatten()

        return pocket_mask

    @staticmethod
    def from_pdb(
        pdb_files: List[PathLike],
        ligand_sdfs: Optional[List[PathLike]] = None,
        pocket_cutoff: float = 7.0,
        remove_ligand_hydrogens: bool = False,
        select_classes: Optional[List[str]] = None,
        remove_water: bool = True,
    ) -> "ProteinData":
        if select_classes is None:
            select_classes = ["ATOM"]
        proteins = ProteinData(pdbs=[PandasPdb().read_pdb(pdb) for pdb in pdb_files])
        if ligand_sdfs is not None:
            mols = [
                Chem.SDMolSupplier(
                    str(ligand_sdfs[i]),
                    removeHs=remove_ligand_hydrogens,
                    sanitize=False,
                )[0]
                for i in range(len(proteins))
            ]
            proteins.apply_pocket_masks(
                mols, pocket_cutoff, select_classes, remove_water
            )
        return proteins

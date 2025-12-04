import tempfile
from itertools import zip_longest
from typing import Optional

import numpy as np
import torch

try:
    from openbabel import openbabel
except:
    pass
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFHasAllMoleculeParams, UFFOptimizeMolecule
from rdkit.Geometry import Point3D
from torch import Tensor
from torch_geometric.typing import OptTensor

from e3mol.experiments.data.datainfo import ADDITIONAL_FEATS_MAP, BOND_LIST, DatasetInfo


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


class Molecule:
    def __init__(
        self,
        atom_types: Tensor,
        positions: Tensor,
        dataset_info: DatasetInfo,
        charges: OptTensor = None,
        bond_types: OptTensor = None,
        rdkit_mol: Optional[Chem.Mol] = None,
        hybridization: OptTensor = None,
        num_Hs: OptTensor = None,
        relax_mol: bool = False,
        max_relax_iter: int = 200,
        sanitize: bool = False,
        check_validity: bool = False,
        build_obabel_mol: bool = False,
        strict: bool = False,
    ):
        """
        atom_types: n      LongTensor
        charges: n         LongTensor
        bond_types: n x n  LongTensor
        positions: n x 3   FloatTensor
        atom_decoder: extracted from dataset_infos.
        """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, (
            f"shape of atoms {atom_types.shape} " f"and dtype {atom_types.dtype}"
        )
        if bond_types is not None:
            assert bond_types.dim() == 2 and bond_types.dtype == torch.long, (
                f"shape of bonds {bond_types.shape} --" f" {bond_types.dtype}"
            )
            assert len(bond_types.shape) == 2

        assert len(atom_types.shape) == 1
        assert len(positions.shape) == 2

        self.relax_mol = relax_mol
        self.max_relax_iter = max_relax_iter
        self.sanitize = sanitize
        self.check_validity = check_validity

        self.dataset_info = dataset_info
        self.atom_decoder = (
            dataset_info["atom_decoder"]
            if isinstance(dataset_info, dict)
            else self.dataset_info.atom_decoder
        )

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long() if bond_types is not None else None
        self.positions = positions
        self.charges = charges

        if isinstance(hybridization, torch.Tensor):
            assert len(hybridization.shape) == 1
            assert (
                hybridization.max().item()
                <= len(ADDITIONAL_FEATS_MAP["hybridization"]) - 1
            )
            self.hybridization = hybridization
        else:
            self.hybridization = []

        if isinstance(num_Hs, torch.Tensor):
            assert len(num_Hs.shape) == 1
            assert num_Hs.max().item() <= len(ADDITIONAL_FEATS_MAP["numHs"]) - 1
            self.num_Hs = num_Hs
        else:
            self.num_Hs = []

        if rdkit_mol is None:
            self.rdkit_mol = (
                self.build_molecule_openbabel()
                if build_obabel_mol or bond_types is None
                else self.build_molecule(strict=strict)
            )
        else:
            self.rdkit_mol = rdkit_mol

        if self.bond_types is None:
            adj = torch.from_numpy(
                Chem.rdmolops.GetAdjacencyMatrix(self.rdkit_mol, useBO=True)
            )
            self.bond_types = adj.long()

        self.num_nodes = len(atom_types)
        self.num_atom_types = len(self.atom_decoder)

    def build_molecule(
        self,
        strict: bool = False,
    ) -> Optional[Chem.Mol]:
        mol = Chem.RWMol()

        for atom, charge, sp_hybridization, num_Hs in zip_longest(
            self.atom_types,
            self.charges,
            self.hybridization,
            self.num_Hs,
            fillvalue=None,
        ):
            if atom == -1:
                continue

            try:
                a = Chem.Atom(self.atom_decoder[int(atom.item())])  # type: ignore[union-attr]
            except Exception:
                continue

            if charge.item() != 0:  # type: ignore[union-attr]
                a.SetFormalCharge(charge.item())  # type: ignore[union-attr]

            if sp_hybridization is not None:
                a.SetHybridization(
                    ADDITIONAL_FEATS_MAP["hybridization"][sp_hybridization.item()]
                )

            if num_Hs is not None:
                a.SetNumExplicitHs(ADDITIONAL_FEATS_MAP["numHs"][num_Hs.item()])

            mol.AddAtom(a)

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for _, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    BOND_LIST[edge_types[bond[0], bond[1]].item()],
                )
        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule. Returning None")
            return None

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    positions[i][0].item(),
                    positions[i][1].item(),
                    positions[i][2].item(),
                ),
            )
        mol.AddConformer(conf)

        if self.relax_mol:
            mol_uff = mol
            try:
                if self.sanitize:
                    Chem.SanitizeMol(mol_uff)
                self.uff_relax(mol_uff, self.max_relax_iter)
                if self.sanitize:
                    Chem.SanitizeMol(mol_uff)
                return mol_uff
            except (RuntimeError, ValueError):
                if self.check_validity:
                    return self.compute_validity(mol)
                else:
                    return mol
        else:
            if self.check_validity:
                return self.compute_validity(mol, strict=strict)
            else:
                return mol

    def build_molecule_openbabel(self) -> Optional[Chem.Mol]:
        """
        Build an RDKit molecule using openbabel for creating bonds
        Args:
            positions: N x 3
            atom_types: N
            atom_decoder: maps indices to atom types
        Returns:
            rdkit molecule if successfully built, else None
        """
        atom_types = [self.atom_decoder[int(a)] for a in self.atom_types]

        try:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp_file = tmp.name

                # Write xyz file
                write_xyz_file(self.positions, atom_types, tmp_file)

                # Convert to sdf file with openbabel
                # openbabel will add bonds
                obConversion = openbabel.OBConversion()
                obConversion.SetInAndOutFormats("xyz", "sdf")
                ob_mol = openbabel.OBMol()
                obConversion.ReadFile(ob_mol, tmp_file)

                obConversion.WriteFile(ob_mol, tmp_file)

                # Read sdf file with RDKit
                tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

            # Build new molecule. This is a workaround to remove radicals.
            mol = Chem.RWMol()
            for atom in tmp_mol.GetAtoms():
                mol.AddAtom(Chem.Atom(atom.GetSymbol()))
            mol.AddConformer(tmp_mol.GetConformer(0))

            for bond in tmp_mol.GetBonds():
                mol.AddBond(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
                )
            mol = self.process_obabel_molecule(
                mol, sanitize=self.sanitize, largest_frag=self.sanitize
            )
        except Exception:
            return None

        return mol

    def process_obabel_molecule(
        self,
        rdmol,
        add_hydrogens=False,
        sanitize=False,
        largest_frag=False,
    ):
        """
        Apply filters to an RDKit molecule. Makes a copy first.
        Args:
            rdmol: rdkit molecule
            add_hydrogens
            sanitize
            relax_iter: maximum number of UFF optimization iterations
            largest_frag: filter out the largest fragment in a set of disjoint
                molecules
        Returns:
            RDKit molecule or None if it does not pass the filters
        """

        # Create a copy
        mol = Chem.Mol(rdmol)

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                print("Sanitization failed. Returning None.")
                return None

        if add_hydrogens:
            mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

        if largest_frag:
            mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if sanitize:
                # sanitize the updated molecule
                try:
                    Chem.SanitizeMol(mol)
                except ValueError:
                    return None

        if self.relax_mol:
            if not UFFHasAllMoleculeParams(mol):
                print("UFF parameters not available for all atoms. " "Returning None.")
                return None

            try:
                self.uff_relax(mol, self.max_relax_iter)
                if sanitize:
                    # sanitize the updated molecule
                    Chem.SanitizeMol(mol)
            except (RuntimeError, ValueError):
                return None

        return mol

    def uff_relax(self, mol, max_iter=200):
        """
        Uses RDKit's universal force field (UFF) implementation to optimize a
        molecule.
        """
        more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
        if more_iterations_required:
            print(
                f"Maximum number of FF iterations reached. "
                f"Returning molecule after {max_iter} relaxation steps."
            )
        return more_iterations_required

    def compute_validity(self, mol, strict=False) -> Optional[Chem.Mol]:
        if mol is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=False
                )
                if len(mol_frags) > 1:
                    return None
                else:
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    if not strict:
                        Chem.SanitizeMol(largest_mol)
                    else:
                        initial_adj = Chem.GetAdjacencyMatrix(
                            largest_mol, useBO=True, force=True
                        )

                        Chem.SanitizeMol(largest_mol)
                        if (
                            sum([a.GetNumImplicitHs() for a in largest_mol.GetAtoms()])
                            > 0
                        ):
                            return None
                        # sanitization changes bond order
                        # without throwing exceptions for certain cases
                        # https://github.com/rdkit/rdkit/blob/
                        # master/Docs/Book/RDKit_Book.rst#molecular-sanitization
                        # only consider change in BO
                        # to be wrong when difference is > 0.5
                        # (not just kekulization difference)
                        adj2 = Chem.GetAdjacencyMatrix(
                            largest_mol, useBO=True, force=True
                        )
                        if not np.all(np.abs(initial_adj - adj2) < 1):
                            return None
                        # atom valencies are only correct when
                        # unpaired electrons are added
                        # when training data does not
                        # contain open shell systems, this should be considered an error
                        if (
                            sum(
                                [
                                    a.GetNumRadicalElectrons()
                                    for a in largest_mol.GetAtoms()
                                ]
                            )
                            > 0
                        ):
                            return None
            except Exception:
                return None
        return mol

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

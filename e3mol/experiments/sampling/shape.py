from typing import Any, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist


def wrapper_shape_tanimoto_distance(
    mol: Chem.Mol,
    mol_ref: Chem.Mol,
    grid_spacing: float = 0.5,
    align_mols: bool = False,
) -> Any | float:
    """
    Wraps the shape tanimoto calculation for a reference molecule (mol_ref)
    to another molecule (mol).
    By default, no alignment is performed.
    If alignment should be performed, set align_mols to True.

    Args:
        mol_ref (rdMol): Reference RDKit molecule
        mol (rdMol): RDKit molecule to compare to the reference
        grid_spacing (float, optional): _description_. Defaults to 0.5.
        align_mols (bool, optional): _description_. Defaults to False.

    Returns:
        float: Shape tanimoto distance between mol_ref and mol
    """
    if align_mols:
        o3d = rdMolAlign.GetO3A(mol, mol_ref)
        o3d.Align()
    return ShapeTanimotoDist(mol_ref, mol, gridSpacing=grid_spacing)


class ShapeTanimotoDistance:
    def __init__(
        self,
        grid_spacing: float = 0.5,
        align_mols: bool = False,
    ) -> None:
        self.grid_spacing = grid_spacing
        self.align_mols = align_mols

    def __call__(self, mols: List[Chem.Mol], ref_mols: List[Chem.Mol]) -> np.ndarray:
        shapesim = []
        for mol in mols:
            shape = [
                wrapper_shape_tanimoto_distance(
                    mol,
                    mol_ref,
                    grid_spacing=self.grid_spacing,
                    align_mols=self.align_mols,
                )
                for mol_ref in ref_mols
            ]

            shapesim.append(shape)
        return np.array(shapesim)

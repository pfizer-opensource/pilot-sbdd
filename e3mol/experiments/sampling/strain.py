from copy import deepcopy

import numpy as np
import rdkit.Chem as Chem
from datamol.conformers._conformers import _get_ff
from rdkit.Chem import AllChem


def calculate_energy(mol: Chem.Mol, forcefield: str = "UFF", add_hs: bool = True):
    """
    Evaluates the energy of a molecule using a force field.

    Args:
        mol: RDKit Mol object representing the molecule.
        forcefield: Force field to use for energy calculation (default: "UFF").
        add_hs: Whether to add hydrogens to the molecule (default: True).

    Returns:
        energy: Calculated energy of the molecule.
                Returns NaN if energy calculation fails.
    """
    mol = Chem.Mol(mol)  # Make a deep copy of the molecule

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = _get_ff(mol, forcefield=forcefield)
    except Exception:
        return np.nan

    try:
        energy = ff.CalcEnergy()
    except Exception as e:
        print(e)
        return np.nan

    return energy


def relax_constrained(
    mol: Chem.Mol, forcefield: str = "UFF", add_hs: bool = True, maxDispl=0.1
):
    """
    Calculates the energy of a molecule using a force field.

    Args:
        mol: RDKit Mol object representing the molecule.
        forcefield: Force field to use for energy calculation (default: "UFF").
        add_hs: Whether to add hydrogens to the molecule (default: True).

    Returns:
        energy: Calculated energy of the molecule (rounded to 2 decimal places).
                Returns NaN if energy calculation fails.
    """
    mol = deepcopy(mol)  # Make a deep copy of the molecule

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = _get_ff(mol, forcefield=forcefield)
    except Exception as e:
        print(e)
        return np.nan

    for i in range(mol.GetNumAtoms()):
        if forcefield == "UFF":
            ff.UFFAddPositionConstraint(i, maxDispl=maxDispl, forceConstant=1.0e5)
        elif forcefield == "MMFF94s":
            ff.MMFFAddPositionConstraint(i, maxDispl=maxDispl, forceConstant=1.0e5)
        else:
            raise ValueError(f"Unsupported force field: {forcefield}")
    try:
        ff.Minimize()
        return mol
    except Exception as e:
        print(e)
        return None


def relax_global(mol: Chem.Mol, forcefield: str = "UFF") -> Chem.Mol | None:
    """Relax a molecule by adding hydrogens, embedding it, and optimizing it
    using the UFF force field.

    Args:
        mol (Chem.Mol): The molecule to relax.

    Returns:
        Chem.Mol: The relaxed molecule.
    """

    # if the molecule is None, return None
    if mol is None:
        return None

    # Incase ring info is not present
    Chem.GetSSSR(mol)  # SSSR: Smallest Set of Smallest Rings

    # make a copy of the molecule
    mol = deepcopy(mol)

    # add hydrogens
    mol = Chem.AddHs(mol, addCoords=True)

    # embed the molecule
    # AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.EmbedMolecule(mol)

    # optimize the molecule
    try:
        if forcefield == "UFF":
            AllChem.UFFOptimizeMolecule(mol)
        elif forcefield == "MMFF94s":
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            raise ValueError(f"Unsupported force field: {forcefield}")
    except Exception as e:
        print(e)
        return None

    # return the molecule
    return mol


def relax_global_on_pose(mol: Chem.Mol, forcefield: str = "UFF") -> Chem.Mol | None:
    """Relax the given pose without position constraints by adding hydrogens and optimizing it
    using the UFF force field.

    Args:
        mol (Chem.Mol): The molecule to relax.

    Returns:
        Chem.Mol: The relaxed molecule.
    """

    # if the molecule is None, return None
    if mol is None:
        return None

    # Incase ring info is not present
    Chem.GetSSSR(mol)  # SSSR: Smallest Set of Smallest Rings

    # make a copy of the molecule
    mol = deepcopy(mol)

    # add hydrogens
    mol = Chem.AddHs(mol, addCoords=True)

    # optimize the molecule
    try:
        if forcefield == "UFF":
            AllChem.UFFOptimizeMolecule(mol)
        elif forcefield == "MMFF94s":
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            raise ValueError(f"Unsupported force field: {forcefield}")
    except Exception as e:
        print(e)
        return None

    return mol


def calculate_strain_energy(
    mol: Chem.Mol, maxDispl: float = 0.1, num_confs: int = 50, forcefield: str = "UFF"
):
    """Calculate the strain energy of a molecule.

    In order to evaluate the global strain energy of a molecule,
    rather than local imperfections
    in bonds distances and angles, we first perform a
    local relaxation of the molecule (by minimizing and allowing
    a small displacement of the atoms) and then
    sample and minimize n conformers of the molecule.

    Args:
        mol (Chem.Mol): The molecule to calculate the strain energy for.
        maxDispl (float): The maximum displacement for position constraints
        during local relaxation. (Default: 0.1)
        num_confs (int): The number of conformers to generate for global relaxation.
        forcefield (str): The force field to use for energy calculations. (Default: "UFF")

    Returns:
        float: The calculated strain energy, or None if the calculation fails.
    """
    try:
        # relax molecule enforcing constraints on the atom positions
        locally_relaxed = relax_constrained(
            mol, maxDispl=maxDispl, forcefield=forcefield
        )
        # sample and minimize n conformers
        global_relaxed = [
            relax_global(mol, forcefield=forcefield) for i in range(num_confs)
        ]
        # alleviate insufficient sampling
        global_relaxed.append(relax_global_on_pose(mol, forcefield=forcefield))

        # calculate the energy of the locally relaxed molecule
        local_energy = calculate_energy(locally_relaxed)

        # calculate the energy of the globally relaxed molecules and take the minimum
        energies = [
            calculate_energy(mol, forcefield=forcefield)
            for mol in global_relaxed
            if mol is not None
        ]
        valid_energies = [e for e in energies if e is not None and not np.isnan(e)]
        if not valid_energies:
            raise ValueError("No valid energies found for global relaxation.")
        global_energy = min(valid_energies)

        # calculate the strain energy
        if local_energy is not None and not np.isnan(local_energy):
            strain_energy = local_energy - global_energy
        else:
            print("Local energy calculation failed or returned NaN.")
            strain_energy = np.nan

        return strain_energy

    except Exception as e:
        print("Warning: Strain energy calculation failed")
        print(e)
        return None

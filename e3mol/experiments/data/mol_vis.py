import ase
import py3Dmol
from rdkit import Chem
from rdkit.Chem import rdMolAlign


def create_ase_mol(elements, positions):
    return ase.Atoms(symbols=elements, positions=positions)


def write_ase_trajectory(ase_mols: list[ase.Atoms], filename: str):
    ase.io.write(filename, ase_mols)


def show_atom_number(mol: Chem.Mol, label: str = "atomLabel") -> Chem.Mol:
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol


def view_3d(mol: Chem.Mol) -> py3Dmol.view:
    view = py3Dmol.view(
        data=Chem.MolToMolBlock(mol),  # Convert the RDKit molecule for py3Dmol
        style={"stick": {}, "sphere": {"scale": 0.1}},
    )
    return view.zoomTo()


def visualize_multiple_3d_mols(
    mol_list: list[Chem.Mol], align_to_first=True, colors=None
) -> py3Dmol.view:
    """
    Visualize multiple 3D RDKit molecules in one py3Dmol viewer

    Args:
        mol_list: List of RDKit molecules with 3D conformers
        align_to_first: Whether to align all molecules to the first one
        colors: List of colors for each molecule (optional)
    """
    if not mol_list:
        return None

    # Default colors if not provided
    if colors is None:
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "yellow",
            "cyan",
            "magenta",
        ]
        colors = colors * (len(mol_list) // len(colors) + 1)  # Repeat if needed

    # Align molecules to first one if requested
    aligned_mols = [mol_list[0]]  # First molecule as reference

    if align_to_first and len(mol_list) > 1:
        ref_mol = mol_list[0]
        for mol in mol_list[1:]:
            try:
                # Align molecule to reference
                aligned_mol = Chem.Mol(mol)
                rdMolAlign.AlignMol(aligned_mol, ref_mol)
                aligned_mols.append(aligned_mol)
            except:
                # If alignment fails, use original molecule
                aligned_mols.append(mol)
    else:
        aligned_mols = mol_list

    # Create py3Dmol viewer
    viewer = py3Dmol.view(width=800, height=600)

    # Add each molecule to the viewer
    for i, mol in enumerate(aligned_mols):
        if mol.GetNumConformers() > 0:
            mol_block = Chem.MolToMolBlock(mol)
            viewer.addModel(mol_block, "mol")
            viewer.setStyle(
                {"model": i},
                {
                    "stick": {"colorscheme": colors[i % len(colors)], "radius": 0.15},
                    "sphere": {"colorscheme": colors[i % len(colors)], "radius": 0.3},
                },
            )

    viewer.zoomTo()
    return viewer

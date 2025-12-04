from typing import List, Tuple

from rdkit import Chem


def cut_on_bonds(
    mol: Chem.Mol, atom_pairs: List[Tuple | List], addDummies=False
) -> Chem.Mol:
    assert len(atom_pairs[0]) == 2
    bonds = [mol.GetBondBetweenAtoms(a[0], a[1]).GetIdx() for a in atom_pairs]
    new_mol = Chem.FragmentOnBonds(
        mol,
        bonds,
        addDummies=addDummies,
    )
    return new_mol


def extract_fragment_with_coords(mol, atom_ids, extract_coords: bool = True):
    # Create an editable molecule
    emol = Chem.EditableMol(Chem.Mol())

    # Add the selected atoms to the editable molecule
    for atom_id in atom_ids:
        atom = mol.GetAtomWithIdx(atom_id)
        emol.AddAtom(atom)

    # Add bonds between the selected atoms
    for bond in mol.GetBonds():
        begin_atom_id = bond.GetBeginAtomIdx()
        end_atom_id = bond.GetEndAtomIdx()
        if begin_atom_id in atom_ids and end_atom_id in atom_ids:
            emol.AddBond(
                atom_ids.index(begin_atom_id),
                atom_ids.index(end_atom_id),
                bond.GetBondType(),
            )

    # Get the new molecule
    fragment = emol.GetMol()

    if extract_coords:
        # Copy 3D coordinates
        conf = mol.GetConformer()
        new_conf = Chem.Conformer(len(atom_ids))
        for i, atom_id in enumerate(atom_ids):
            pos = conf.GetAtomPosition(atom_id)
            new_conf.SetAtomPosition(i, pos)
        fragment.AddConformer(new_conf)

    try:
        Chem.SanitizeMol(fragment)
    except Exception:
        pass

    return fragment

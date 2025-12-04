from typing import Any, Dict, List

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength, GetDihedralDeg
from tqdm import tqdm

# Define bond types mapping
BOND_TYPES = {
    Chem.BondType.SINGLE: "-",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: ":",
    Chem.BondType.AROMATIC: "|",
}

TARGET_BONDS = [
    "C=N",
    "C-O",
    "C=O",
    "C=S",
    "N:N",
    "O-O",
    "C:C",
    "C-F",
    "C:N",
    "N-N",
    "C-C",
    "N-O",
    "C=C",
    "C-N",
    "C-Cl",
    "N=N",
    "N=O",
    "Br-C",
    "C-S",
    "C-I",
    "C|C",
    "C|N",
    "C|S",
    "C|O",
]


def get_bond_distances(
    mol: Chem.Mol, target_bonds: List[str] | None = None
) -> List[Dict[str, float]]:

    if target_bonds is None:
        target_bonds = TARGET_BONDS

    conf = mol.GetConformer()
    bond_distances = list()
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        atom1_symbol = atom1.GetSymbol()
        atom2_symbol = atom2.GetSymbol()
        bond_type = bond.GetBondType()

        bond_symbol = BOND_TYPES.get(bond_type, "?")
        s0 = atom1_symbol + bond_symbol + atom2_symbol
        s1 = atom2_symbol + bond_symbol + atom1_symbol

        if s0 in target_bonds or s1 in target_bonds:
            bond_length = GetBondLength(conf, atom1.GetIdx(), atom2.GetIdx())
            at12 = sorted([atom1_symbol, atom2_symbol])
            bond_key = f"{at12[0]}{bond_symbol}{at12[1]}"
            bond_distances.append({bond_key: bond_length})
    return bond_distances


def get_bond_angles(mol: Chem.Mol) -> List[Dict[str, float]]:
    conf = mol.GetConformer()
    angle_values = list()
    for atom in mol.GetAtoms():
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                atom1, atom2, atom3 = neighbors[i], atom.GetIdx(), neighbors[j]
                elem1, elem2, elem3 = (
                    mol.GetAtomWithIdx(atom1).GetSymbol(),
                    atom.GetSymbol(),
                    mol.GetAtomWithIdx(atom3).GetSymbol(),
                )
                bond1 = mol.GetBondBetweenAtoms(atom1, atom2)
                bond2 = mol.GetBondBetweenAtoms(atom2, atom3)

                if bond1 and bond2:
                    bond_symbol1 = BOND_TYPES.get(bond1.GetBondType(), "?")
                    bond_symbol2 = BOND_TYPES.get(bond2.GetBondType(), "?")

                    atoms_and_bonds = sorted(
                        [(elem1, bond_symbol1), (elem3, bond_symbol2)]
                    )
                    sorted_elem1, sorted_bond1 = atoms_and_bonds[0]
                    sorted_elem3, sorted_bond2 = atoms_and_bonds[1]

                    angle = GetAngleDeg(conf, atom1, atom2, atom3)
                    angle_key = f"{sorted_elem1}{sorted_bond1}{elem2}{sorted_bond2}{sorted_elem3}"  # noqa: 950
                    angle_values.append({angle_key: angle})
    return angle_values


def get_dihedral_angles(mol: Chem.Mol):

    dihedrals = []
    try:
        conf = mol.GetConformer()
    except ValueError:
        return None

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != rdchem.BondType.SINGLE:
            continue

        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        if begin_atom.GetDegree() < 2 or end_atom.GetDegree() < 2:
            continue

        begin_neighbors = [
            nbr.GetIdx()
            for nbr in begin_atom.GetNeighbors()
            if nbr.GetIdx() != end_atom.GetIdx()
        ]
        end_neighbors = [
            nbr.GetIdx()
            for nbr in end_atom.GetNeighbors()
            if nbr.GetIdx() != begin_atom.GetIdx()
        ]

        if not begin_neighbors or not end_neighbors:
            continue

        idx1 = begin_neighbors[0]
        idx2 = begin_atom.GetIdx()
        idx3 = end_atom.GetIdx()
        idx4 = end_neighbors[0]
        try:
            angle = GetDihedralDeg(conf, idx1, idx2, idx3, idx4)
            dihedrals.append(angle)
        except Exception:
            continue
    return dihedrals


def wrap_removeHs_sanitize(mols: List[Chem.Mol]) -> List[Chem.Mol]:
    out_mols = []
    for mol in mols:
        mol = Chem.RemoveAllHs(mol)
        Chem.SanitizeMol(mol)
        out_mols.append(mol)
    return out_mols


def process_bond_distances_mols(
    mol_list: List[Chem.Mol],
    target_bonds: List[str] | None = None,
    verbose: bool = False,
    removeHs: bool = True,
) -> Dict[str, List[float]]:

    if target_bonds is None:
        target_bonds = TARGET_BONDS
    if removeHs:
        mol_list = wrap_removeHs_sanitize(mol_list)
    bond_dictionary: Dict[str, List] = {bond_key: [] for bond_key in target_bonds}
    iterator = tqdm(mol_list, total=len(mol_list)) if verbose else mol_list
    for mol in iterator:
        bond_list = get_bond_distances(mol, target_bonds)
        for bond in bond_list:
            assert len(bond) == 1
            bond_dictionary[list(bond.keys())[0]].append(list(bond.values())[0])

    return bond_dictionary


def process_bond_angles_mols(
    mol_list: List[Chem.Mol],
    verbose: bool = False,
    removeHs: bool = True,
) -> Dict[str, List[float]]:
    angles_dictionary: Dict[str, Any] = {}
    if removeHs:
        mol_list = wrap_removeHs_sanitize(mol_list)
    iterator = tqdm(mol_list, total=len(mol_list)) if verbose else mol_list
    for mol in iterator:
        angles_list = get_bond_angles(mol)
        for angle in angles_list:
            assert len(angle) == 1
            k, v = list(angle.keys())[0], list(angle.values())[0]
            if k in angles_dictionary:
                angles_dictionary[k].append(v)
            else:
                angles_dictionary[k] = [v]
    return angles_dictionary

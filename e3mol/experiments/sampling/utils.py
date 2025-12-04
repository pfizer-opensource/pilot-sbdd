import logging
import math
from collections import Counter
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from e3mol.experiments.data.molecule import Molecule

# Mute RDKit logger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}

allowed_bonds_nocharges = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def split_list(data, num_chunks):
    """
    Splits 'data' into 'num_chunks' sublists, ensuring each chunk has at least 2 elements.

    Args:
        data: List to be split
        num_chunks: Number of chunks to split into

    Returns:
        List of sublists, each containing at least 2 elements where possible
    """
    # Adjust num_chunks if there aren't enough elements for 2 per chunk
    total_elements = len(data)
    max_possible_chunks = total_elements // 2
    num_chunks = min(num_chunks, max_possible_chunks)

    if num_chunks <= 0:
        return [data]

    # Calculate base size and remainder
    chunk_size = total_elements // num_chunks
    remainder = total_elements % num_chunks

    # Ensure minimum of 2 elements per chunk
    if chunk_size < 2:
        chunk_size = 2

    chunks = []
    start = 0
    for i in range(num_chunks):
        # Add one extra element from remainder if available
        extra = 1 if i < remainder else 0
        chunk_end = start + chunk_size + extra

        # Handle last chunk specially to include any remaining elements
        if i == num_chunks - 1:
            chunk_end = len(data)

        chunks.append(data[start:chunk_end])
        start = chunk_end

        # Break if we've used all elements
        if start >= len(data):
            break

    return chunks


def get_random_subset(
    dataset: List[Any], subset_size: int, seed: Optional[int] = None
) -> List[Any]:
    if len(dataset) < subset_size:
        raise Exception(
            f"The dataset to extract a subset from is too small: "
            f"{len(dataset)} < {subset_size}"
        )

    # save random number generator state
    rng_state = np.random.get_state()

    if seed is not None:
        # extract a subset (for a given training set, the subset will always be identical).
        np.random.seed(seed)

    subset = np.random.choice(dataset, subset_size, replace=False)

    if seed is not None:
        # reset random number generator state, only if needed
        np.random.set_state(rng_state)

    return list(subset)


def check_stability(
    molecule: Molecule,
    atom_decoder: dict,
    debug=False,
    smiles=None,
):
    atom_types = molecule.atom_types
    edge_types = molecule.bond_types

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for _, (atom_type, valency, charge) in enumerate(
        zip(atom_types, valencies, molecule.charges)
    ):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == valency
        elif isinstance(possible_bonds, dict):
            expected_bonds = (
                possible_bonds[charge]
                if charge in possible_bonds.keys()
                else possible_bonds[0]
            )
            is_stable = (
                expected_bonds == valency
                if isinstance(expected_bonds, int)
                else valency in expected_bonds
            )
        else:
            is_stable = valency in possible_bonds  # type: ignore[operator]
        if not is_stable:
            mol_stable = False
        if not is_stable and debug:
            if smiles is not None:
                print(smiles)
            print(
                f"Invalid atom {atom_decoder[atom_type]}: valency={valency}, charge={charge}"
            )
            print()
        n_stable_bonds += int(is_stable)

    return float(mol_stable), n_stable_bonds, len(atom_types)


def canonicalize(
    smiles: str, include_stereocenters=True, remove_hs=False
) -> Optional[str] | Any:
    mol = Chem.MolFromSmiles(smiles)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


def canonicalize_list(
    smiles_list,
    include_stereocenters=True,
    remove_hs=False,
):
    canonicalized_smiles = [
        canonicalize(smiles, include_stereocenters, remove_hs=remove_hs)
        for smiles in smiles_list
    ]
    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)


def remove_duplicates(list_with_duplicates):
    unique_set = set()
    unique_list = []
    ids = []
    for i, element in enumerate(list_with_duplicates):
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)
        else:
            ids.append(i)

    return unique_list, ids


def get_mols(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                yield mol
        except Exception as e:
            logger.warning(e)


def get_mols_list(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    mols = []
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                mols.append(mol)
        except Exception as e:
            logger.warning(e)
    return mols


def get_fingerprints_from_smiles_list(smiles_list: List[str]):
    """
    Converts the provided smiles into ECFP4 bitvectors of length 4096.

    Args:
        smiles_list: list of SMILES strings

    Returns: ECFP4 bitvectors of length 4096.

    """
    return get_fingerprints(get_mols(smiles_list))


def get_fingerprints(
    mols: Iterable[Chem.Mol], radius=2, length=4096, chiral=False, sanitize=False
):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=length, includeChirality=chiral
    )
    if sanitize:
        fps = []
        for mol in mols:
            Chem.SanitizeMol(mol)
            fps.append(mfpgen.GetFingerprint(mol))
    else:
        fps = [mfpgen.GetFingerprint(mol) for mol in mols]
    return fps


def wasserstein1d(preds, target, step_size=1):
    """preds and target are 1d tensors.
    They contain histograms for bins that are regularly spaced
    """
    target = normalize(target) / step_size
    preds = normalize(preds) / step_size
    max_len = max(len(preds), len(target))
    preds = F.pad(preds, (0, max_len - len(preds)))
    target = F.pad(target, (0, max_len - len(target)))

    cs_target = torch.cumsum(target, dim=0)
    cs_preds = torch.cumsum(preds, dim=0)
    return torch.sum(torch.abs(cs_preds - cs_target)).item()


def total_variation1d(preds, target):
    assert (
        target.dim() == 1 and preds.shape == target.shape
    ), f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s


def counter_to_tensor(c: Counter):
    max_key = max(c.keys())
    assert isinstance(max_key, int)
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def number_nodes_distance(molecules, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for molecule in molecules:
        c[molecule.num_nodes] += 1

    generated_n = counter_to_tensor(c)
    return wasserstein1d(generated_n, reference_n)


def atom_types_distance(molecules, target, save_histogram=False):
    generated_distribution = torch.zeros_like(target)
    for molecule in molecules:
        for atom_type in molecule.atom_types:
            generated_distribution[atom_type] += 1
    if save_histogram:
        np.save("generated_atom_types.npy", generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def bond_types_distance(molecules, target, save_histogram=False):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        bond_types = molecule.bond_types
        mask = torch.ones_like(bond_types)
        mask = torch.triu(mask, diagonal=1).bool()
        bond_types = bond_types[mask]
        unique_edge_types, counts = torch.unique(bond_types, return_counts=True)
        for type, count in zip(unique_edge_types, counts):
            generated_distribution[type] += count
    if save_histogram:
        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())
    sparsity_level = generated_distribution[0] / torch.sum(generated_distribution)
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class, sparsity_level


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    device = molecules[0].charges.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.atom_types == atom_type
            if mask.sum() > 0:
                at_charges = dataset_infos.one_hot_charges(molecule.charges[mask])
                generated_distribution[atom_type] += at_charges.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities.to(device)).item()

    return w1, w1_per_class


def valency_distance(
    molecules, target_valencies, atom_types_probabilities, atom_encoder
):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(
        max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values()
    )
    max_valency_generated = max(
        max(vals.keys()) if len(vals) > 0 else -1
        for vals in generated_valencies.values()
    )
    max_valency = max(max_valency_target, max_valency_generated)

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[atom_encoder[atom_type], valency] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[atom_type, valency] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


def bond_length_distance(molecules, target, bond_types_probabilities):
    generated_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for molecule in molecules:
        cdists = torch.cdist(
            molecule.positions.unsqueeze(0), molecule.positions.unsqueeze(0)
        ).squeeze(0)
        for bond_type in range(1, 5):
            edges = torch.nonzero(molecule.bond_types == bond_type)
            bond_distances = cdists[edges[:, 0], edges[:, 1]]
            distances_to_consider = torch.round(bond_distances, decimals=2)
            for d in distances_to_consider:
                generated_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, 5):
        s = sum(generated_bond_lenghts[bond_type].values())
        if s == 0:
            s = 1
        for d, count in generated_bond_lenghts[bond_type].items():
            generated_bond_lenghts[bond_type][d] = count / s

    # Convert both dictionaries to tensors
    min_generated_length = min(
        min(d.keys()) if len(d) > 0 else 1e4 for d in generated_bond_lenghts.values()
    )
    min_target_length = min(
        min(d.keys()) if len(d) > 0 else 1e4 for d in target.values()
    )
    min_length = min(min_generated_length, min_target_length)

    max_generated_length = max(
        max(bl.keys()) if len(bl) > 0 else -1 for bl in generated_bond_lenghts.values()
    )
    max_target_length = max(
        max(bl.keys()) if len(bl) > 0 else -1 for bl in target.values()
    )
    max_length = max(max_generated_length, max_target_length)

    num_bins = int((max_length - min_length) * 100) + 1
    generated_bond_lengths = torch.zeros(4, num_bins)
    target_bond_lengths = torch.zeros(4, num_bins)

    for bond_type in range(1, 5):
        for d, count in generated_bond_lenghts[bond_type].items():
            bin = int((d - min_length) * 100)
            generated_bond_lengths[bond_type - 1, bin] = count
        for d, count in target[bond_type].items():
            bin = int((d - min_length) * 100)
            target_bond_lengths[bond_type - 1, bin] = count

    cs_generated = torch.cumsum(generated_bond_lengths, dim=1)
    cs_target = torch.cumsum(target_bond_lengths, dim=1)

    w1_per_class = (
        torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 100
    )  # 100 because of bin size
    weighted = w1_per_class * bond_types_probabilities[1:]
    return torch.sum(weighted).item(), w1_per_class


def angle_distance(
    molecules,
    target_angles,
    atom_types_probabilities,
    valencies,
    atom_decoder,
    save_histogram: bool,
):
    num_atom_types = len(atom_types_probabilities)
    generated_angles = torch.zeros(num_atom_types, 180 * 10 + 1)
    for molecule in molecules:
        adj = molecule.bond_types
        pos = molecule.positions
        for atom in range(adj.shape[0]):
            p_a = pos[atom]
            neighbors = torch.nonzero(adj[atom]).squeeze(1)
            for i in range(len(neighbors)):
                p_i = pos[neighbors[i]]
                for j in range(i + 1, len(neighbors)):
                    p_j = pos[neighbors[j]]
                    v1 = p_i - p_a
                    v2 = p_j - p_a
                    assert not torch.isnan(v1).any()
                    assert not torch.isnan(v2).any()
                    prod = torch.dot(
                        v1 / (torch.norm(v1) + 1e-6), v2 / (torch.norm(v2) + 1e-6)
                    )
                    if prod > 1:
                        print(
                            f"Invalid angle {i} {j} -- {prod} \
                            -- {v1 / (torch.norm(v1) + 1e-6)} --"
                            f" {v2 / (torch.norm(v2) + 1e-6)}"
                        )
                    prod.clamp(min=0, max=1)
                    angle = torch.acos(prod)
                    if torch.isnan(angle).any():
                        print(
                            f"Nan obtained in angle {i} {j} \
                            -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                            f" {v2 / (torch.norm(v2) + 1e-6)}"
                        )
                    else:
                        bin = int(
                            torch.round(angle * 180 / math.pi, decimals=1).item() * 10
                        )
                        generated_angles[molecule.atom_types[atom], bin] += 1

    s = torch.sum(generated_angles, dim=1, keepdim=True)
    s[s == 0] = 1
    generated_angles = generated_angles / s
    if save_histogram:
        np.save("generated_angles_histogram.npy", generated_angles.numpy())

    if type(target_angles) in [np.array, np.ndarray]:
        target_angles = torch.from_numpy(target_angles).float()

    cs_generated = torch.cumsum(generated_angles, dim=1)
    cs_target = torch.cumsum(target_angles, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    # The atoms that have a valency less than 2 should not matter
    valency_weight = torch.zeros(len(w1_per_type), device=w1_per_type.device)
    for i in range(len(w1_per_type)):
        valency_weight[i] = (
            1 - valencies[atom_decoder[i]][0] - valencies[atom_decoder[i]][1]
        )

    weighted = w1_per_type * atom_types_probabilities * valency_weight
    return (
        torch.sum(weighted)
        / (torch.sum(atom_types_probabilities * valency_weight) + 1e-5)
    ).item(), w1_per_type


def dihedral_distance(
    molecules,
    target_dihedrals,
    bond_types_probabilities,
    save_histogram,
):
    def calculate_dihedral_angles(mol):
        def find_dihedrals(mol):
            torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
            torsionQuery = Chem.MolFromSmarts(torsionSmarts)
            matches = mol.GetSubstructMatches(torsionQuery)
            torsionList = []
            btype = []
            for match in matches:
                idx2 = match[0]
                idx3 = match[1]
                bond = mol.GetBondBetweenAtoms(idx2, idx3)
                jAtom = mol.GetAtomWithIdx(idx2)
                kAtom = mol.GetAtomWithIdx(idx3)
                if (
                    (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ) or (
                    (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ):
                    continue
                for b1 in jAtom.GetBonds():
                    if b1.GetIdx() == bond.GetIdx():
                        continue
                    idx1 = b1.GetOtherAtomIdx(idx2)
                    for b2 in kAtom.GetBonds():
                        if (b2.GetIdx() == bond.GetIdx()) or (
                            b2.GetIdx() == b1.GetIdx()
                        ):
                            continue
                        idx4 = b2.GetOtherAtomIdx(idx3)
                        # skip 3-membered rings
                        if idx4 == idx1:
                            continue
                        bt = bond.GetBondTypeAsDouble()
                        # bt = str(bond.GetBondType())
                        # if bond.IsInRing():
                        #     bt += '_R'
                        btype.append(bt)
                        torsionList.append((idx1, idx2, idx3, idx4))
            return np.asarray(torsionList), np.asarray(btype)

        dihedral_idx, dihedral_types = find_dihedrals(mol)

        coords = mol.GetConformer().GetPositions()
        t_angles = []
        for t in dihedral_idx:
            u1, u2, u3, u4 = coords[torch.tensor(t)]

            a1 = u2 - u1
            a2 = u3 - u2
            a3 = u4 - u3

            v1 = np.cross(a1, a2)
            v1 = v1 / (v1 * v1).sum(-1) ** 0.5
            v2 = np.cross(a2, a3)
            v2 = v2 / (v2 * v2).sum(-1) ** 0.5
            porm = np.sign((v1 * a3).sum(-1))
            rad = np.arccos(
                (v1 * v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1) + 1e-9) ** 0.5
            )
            if not porm == 0:
                rad = rad * porm
            t_angles.append(rad * 180 / torch.pi)

        return np.asarray(t_angles), dihedral_types

    # forget about none and tripple bonds
    bond_types_probabilities[torch.tensor([0, 3])] = 0
    bond_types_probabilities /= bond_types_probabilities.sum()

    num_bond_types = len(bond_types_probabilities)
    generated_dihedrals = torch.zeros(num_bond_types, 180 * 10 + 1)
    for mol in molecules:
        mol = mol.rdkit_mol
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        angles, types = calculate_dihedral_angles(mol)
        # transform types to idx
        types[types == 1.5] = 4
        types = types.astype(int)
        for a, t in zip(np.abs(angles), types):
            if np.isnan(a):
                continue
            generated_dihedrals[t, int(np.round(a, decimals=1) * 10)] += 1

    # normalize
    s = generated_dihedrals.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    generated_dihedrals = generated_dihedrals.float() / s

    if save_histogram:
        np.save("generated_dihedrals_historgram.npy", generated_dihedrals.numpy())

    if type(target_dihedrals) in [np.array, np.ndarray]:
        target_dihedrals = torch.from_numpy(target_dihedrals).float()

    cs_generated = torch.cumsum(generated_dihedrals, dim=1)
    cs_target = torch.cumsum(target_dihedrals, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    weighted = w1_per_type * bond_types_probabilities

    return torch.sum(weighted).item(), w1_per_type

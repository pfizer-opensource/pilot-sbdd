import math
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from e3mol.experiments.data.datainfo import ADDITIONAL_FEATS_MAP as x_map
from e3mol.experiments.data.datainfo import Statistics


def node_counts(data_list: List[Data]) -> Counter:
    print("Computing number of nodes counts...")
    all_node_counts: Counter = Counter()
    for data in data_list:
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def atom_type_counts(
    data_list: List[Data], num_classes: int, normalize: bool = True
) -> np.ndarray:
    print("Computing node types distribution...")
    counts = np.zeros(num_classes)
    for data in data_list:
        x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
        counts += x.sum(dim=0).numpy()
    if normalize:
        counts = counts / counts.sum()
    print("Done.")
    return counts


def edge_counts(
    data_list: List[Data], num_bond_types: int = 5, normalize: bool = True
) -> np.ndarray:
    print("Computing edge types distribution...")
    d = np.zeros(num_bond_types)

    for data in data_list:
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = (
            torch.nn.functional.one_hot(
                data.edge_attr - 1, num_classes=num_bond_types - 1
            )
            .sum(dim=0)
            .numpy()
        )
        d[0] += num_non_edges
        d[1:] += edge_types
    if normalize:
        d = d / d.sum()
    return d


def charge_counts(
    data_list: List[Data], num_classes: int, charges_dic: dict, normalize: bool = True
) -> np.ndarray:
    print("Computing charge counts...")
    d = np.zeros((num_classes, len(charges_dic)))

    for data in data_list:
        for atom, charge in zip(data.x, data.charges):
            assert charge in [-2, -1, 0, 1, 2, 3]
            d[atom.item(), charges_dic[charge.item()]] += 1

    s = np.sum(d, axis=1, keepdims=True)
    s[s == 0] = 1
    if normalize:
        d = d / s
    print("Done.")
    return d


def valency_count(
    data_list: List[Data], atom_encoder: dict, normalize=True
) -> Dict[str, Counter]:
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing valency counts...")
    valencies: Dict[str, Counter] = {
        atom_type: Counter() for atom_type in atom_encoder.keys()
    }

    for data in data_list:
        edge_attr = data.edge_attr
        edge_attr[edge_attr == 4] = 1.5
        bond_orders = edge_attr

        for atom in range(data.num_nodes):
            edges = bond_orders[data.edge_index[0] == atom]
            valency = edges.sum(dim=0)
            valencies[atom_decoder[data.x[atom].item()]][valency.item()] += 1

    if normalize:
        # Normalizing the valency counts
        for atom_type in valencies.keys():
            s = sum(valencies[atom_type].values())
            for valency, count in valencies[atom_type].items():
                valencies[atom_type][valency] = count / s  # type: ignore
    print("Done.")
    return valencies


def additional_feat_counts(
    data_list,
    keys: Optional[List] = None,
    normalize=True,
) -> dict:

    if keys is None:
        keys = [
            "is_aromatic",
            "is_in_ring",
            "hybridization",
            "is_h_donor",
            "is_h_acceptor",
            "numHs",
        ]

    print(f"Computing node counts for features = {str(keys)}")

    num_classes_list = [len(x_map[key]) for key in keys]
    counts_list = [np.zeros(num_classes) for num_classes in num_classes_list]

    for data in data_list:
        for i, key, num_classes in zip(range(len(keys)), keys, num_classes_list):
            x = torch.nn.functional.one_hot(data.get(key), num_classes=num_classes)
            counts_list[i] += x.sum(dim=0).numpy()

    if normalize:
        for i in range(len(counts_list)):
            counts_list[i] = counts_list[i] / counts_list[i].sum()
    print("Done")

    results = dict()
    for key, count in zip(keys, counts_list):
        results[key] = count

    print(results)

    return results


def counter_to_tensor(c: Counter) -> torch.Tensor:
    max_key = max(c.keys())
    assert isinstance(max_key, int)
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def compute_bond_lengths_counts(data_list, num_bond_types=5, normalize=True):
    """Compute the bond lenghts separetely for each bond type."""
    print("Computing bond lengths...")
    all_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for data in data_list:
        cdists = torch.cdist(data.pos.unsqueeze(0), data.pos.unsqueeze(0)).squeeze(0)
        bond_distances = cdists[data.edge_index[0], data.edge_index[1]]
        for bond_type in range(1, num_bond_types):
            bond_type_mask = data.edge_attr == bond_type
            distances_to_consider = bond_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                all_bond_lenghts[bond_type][d.item()] += 1

    if normalize:
        # Normalizing the bond lenghts
        for bond_type in range(1, num_bond_types):
            s = sum(all_bond_lenghts[bond_type].values())
            for d, count in all_bond_lenghts[bond_type].items():
                all_bond_lenghts[bond_type][d] = count / s
    print("Done.")
    return all_bond_lenghts


def compute_bond_angles(data_list, atom_encoder, normalize=True):
    print("Computing bond angles...")
    all_bond_angles = np.zeros((len(atom_encoder.keys()), 180 * 10 + 1))
    for data in data_list:
        assert not torch.isnan(data.pos).any()
        for i in range(data.num_nodes):
            neighbors = data.edge_index[1][data.edge_index[0] == i]
            for j in neighbors:
                for k in neighbors:
                    if j == k:
                        continue
                    assert i != j and i != k and j != k, "i, j, k: {}, {}, {}".format(
                        i, j, k
                    )
                    a = data.pos[j] - data.pos[i]
                    b = data.pos[k] - data.pos[i]

                    # print(a, b, torch.norm(a) * torch.norm(b))
                    angle = torch.acos(
                        torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6)
                    )
                    angle = angle * 180 / math.pi

                    bin = int(torch.round(angle, decimals=1) * 10)
                    all_bond_angles[data.x[i].item(), bin] += 1

    if normalize:
        # Normalizing the angles
        s = all_bond_angles.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        all_bond_angles = all_bond_angles / s
    print("Done.")
    return all_bond_angles


def compute_spatial_cov(data_list: List[Data]) -> Dict[int, float]:
    print("Computing spatial covariance...")
    n_atoms_cov = dict()  # type: Dict[int, float]
    for data in data_list:
        mol = Chem.RemoveHs(data.mol)
        P = mol.GetConformers()[0].GetPositions()
        P = P - np.mean(P, axis=0, keepdims=True)
        N = mol.GetNumAtoms()
        cov = np.cov(P.T)
        v = np.mean(np.diag(cov)).item()
        if N in n_atoms_cov.keys():
            n_atoms_cov[N].extend([v])  # type: ignore[attr-defined]
        else:
            n_atoms_cov[N] = [v]  # type: ignore

    # sort in increasing order
    n_atoms_cov = dict(sorted(n_atoms_cov.items()))
    # select median value as representative for each number of atoms
    n_atoms_cov = {k: np.median(v).item() for k, v in n_atoms_cov.items()}
    print("Done.")
    return n_atoms_cov


def dihedral_angles(data_list, normalize=True):
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

    generated_dihedrals = torch.zeros(5, 180 * 10 + 1)
    for d in data_list:
        mol = d.mol
        angles, types = calculate_dihedral_angles(mol)
        # transform types to idx
        types[types == 1.5] = 4
        types = types.astype(int)
        for a, t in zip(np.abs(angles), types):
            if np.isnan(a):
                continue
            generated_dihedrals[t, int(np.round(a, decimals=1) * 10)] += 1

    if normalize:
        s = generated_dihedrals.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        generated_dihedrals = generated_dihedrals.float() / s

    return generated_dihedrals


def compute_all_statistics(
    data_list: List[Data],
    atom_encoder,
    charges_dic: dict,
    additional_feats: bool = True,
    normalize=True,
    bond_angles_distribution: bool = True,
) -> Statistics:
    num_nodes = node_counts(data_list)
    atom_types = atom_type_counts(
        data_list, num_classes=len(atom_encoder), normalize=normalize
    )
    print(f"Atom types: {atom_types}")
    bond_types = edge_counts(data_list, num_bond_types=5, normalize=normalize)
    print(f"Bond types: {bond_types}")
    charge_types = charge_counts(
        data_list,
        num_classes=len(atom_encoder),
        charges_dic=charges_dic,
        normalize=normalize,
    )
    print(f"Charge types: {charge_types}")
    valency = valency_count(data_list, atom_encoder, normalize=normalize)
    print("Valency: ", valency)

    if bond_angles_distribution:
        bond_lengths = compute_bond_lengths_counts(data_list, normalize=normalize)
        print("Bond lengths: ", bond_lengths)
        bond_angles = compute_bond_angles(data_list, atom_encoder, normalize=normalize)
        print("Bond angles: ", bond_angles)
        dihedrals = dihedral_angles(data_list, normalize=normalize)
        print("Dihedrals: ", dihedrals)
    else:
        bond_lengths = None
        bond_angles = None
        dihedrals = None

    feats = {}
    if additional_feats:
        feats.update(additional_feat_counts(data_list=data_list, normalize=normalize))
    n_atoms_cov = compute_spatial_cov(data_list)
    feats["n_atoms_cov"] = n_atoms_cov
    print("Number of atoms and spatial covariance: ", n_atoms_cov)
    print()

    return Statistics(
        num_nodes=num_nodes,
        atom_types=atom_types,
        bond_types=bond_types,
        charge_types=charge_types,
        valencies=valency,
        bond_lengths=bond_lengths,
        bond_angles=bond_angles,
        dihedrals=dihedrals,
        **feats,
    )

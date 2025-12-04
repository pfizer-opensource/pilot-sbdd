import pickle
from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np
import rdkit
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor

CHARGES_DICT = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5}

ATOM_ENCODER = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Se": 12,
    "Br": 13,
    "I": 14,
    "Hg": 15,
    "Bi": 16,
}
ATOM_DECODER = {v: k for k, v in ATOM_ENCODER.items()}

ELEMENT_TO_ATOMIC_NUM = {k: Chem.Atom(k).GetAtomicNum() for k in ATOM_ENCODER.keys()}
# PT = Chem.GetPeriodicTable()
# ELEMENT_TO_VDW_RADII = {k: PT.GetRvdw(v) for k, v in ELEMENT_TO_ATOMIC_NUM.items()}

ELEMENT_TO_VDW_RADII = {
    "H": 1.2,
    "B": 1.8,
    "C": 1.7,
    "N": 1.6,
    "O": 1.55,
    "F": 1.5,
    "Al": 2.1,
    "Si": 2.1,
    "P": 1.95,
    "S": 1.8,
    "Cl": 1.8,
    "As": 2.05,
    "Se": 1.9,
    "Br": 1.9,
    "I": 2.1,
    "Hg": 2.05,
    "Bi": 2.3,
}


def get_vdw_radius_from_integer(integer):
    return ELEMENT_TO_VDW_RADII[ATOM_DECODER[integer]]


def get_vdw_radius_from_integer_np(arr: np.ndarray):
    assert arr.ndim == 1
    return np.vectorize(get_vdw_radius_from_integer)(arr)


ADDITIONAL_FEATS_MAP = {
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
    "hybridization": [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ],
    "is_h_donor": [False, True],
    "is_h_acceptor": [False, True],
    "numHs": [0, 1, 2, 3, 4],
}


BOND_LIST = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


AA_ENCODER: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

AA_DECODER = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


class DistributionNodes:
    def __init__(self, histogram: Union[Dict[int, int], Tensor]):
        if isinstance(histogram, dict):
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        probas = self.prob[batch_n_nodes.to(self.prob.device)]
        log_p = torch.log(probas + 1e-10)
        return log_p.to(batch_n_nodes.device)


class ConditionalDistributionNodes:
    def __init__(self, histogram: Union[List[List], Tensor]):
        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)

        self.n_nodes_to_idx = {
            tuple(x.tolist()): i for i, x in enumerate(self.idx_to_n_nodes)
        }

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob.view(-1), validate_args=True)

        self.n1_given_n2 = [
            torch.distributions.Categorical(prob[:, j], validate_args=True)
            for j in range(prob.shape[1])
        ]
        self.n2_given_n1 = [
            torch.distributions.Categorical(prob[i, :], validate_args=True)
            for i in range(prob.shape[0])
        ]

        # entropy = -torch.sum(self.prob.view(-1) * torch.log(self.prob.view(-1) + 1e-30))
        # entropy = self.m.entropy()
        # print("Entropy of n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None), "Exactly one input argument must be None"

        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1

        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        assert len(batch_n_nodes_1.size()) == 1
        assert len(batch_n_nodes_2.size()) == 1

        idx = torch.tensor(
            [
                self.n_nodes_to_idx[(n1, n2)]
                for n1, n2 in zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())
            ]
        )

        # log_probs = torch.log(self.prob.view(-1)[idx] + 1e-30)
        log_probs = self.m.log_prob(idx)

        return log_probs.to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        assert len(n1.size()) == 1
        assert len(n2.size()) == 1
        log_probs = torch.stack(
            [self.n1_given_n2[c].log_prob(i.cpu()) for i, c in zip(n1, n2)]
        )
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        assert len(n2.size()) == 1
        assert len(n1.size()) == 1
        log_probs = torch.stack(
            [self.n2_given_n1[c].log_prob(i.cpu()) for i, c in zip(n2, n1)]
        )
        return log_probs.to(n2.device)


class Statistics:
    def __init__(
        self,
        num_nodes: Counter,
        atom_types: Tensor,
        bond_types: Tensor,
        charge_types: Tensor,
        valencies: Dict[str, Counter],
        bond_lengths: Optional[Dict[int, Counter]] = None,
        bond_angles: Optional[Tensor] = None,
        dihedrals: Optional[Tensor] = None,
        is_in_ring: Optional[Tensor] = None,
        is_aromatic: Optional[Tensor] = None,
        hybridization: Optional[Tensor] = None,
        force_norms: Optional[Tensor] = None,
        is_h_donor: Optional[Tensor] = None,
        is_h_acceptor: Optional[Tensor] = None,
        n_atoms_cov: Optional[dict] = None,
        numHs: Optional[Tensor] = None,
        **kwargs,
    ):
        self.num_nodes = num_nodes
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedrals = dihedrals
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.hybridization = hybridization
        self.force_norms = force_norms
        self.is_h_donor = is_h_donor
        self.is_h_acceptor = is_h_acceptor
        self.n_atoms_cov = n_atoms_cov
        self.numHs = numHs

        self.check_tensor()

    def to_dict(self):
        return vars(self)

    def check_tensor(self):
        self.atom_types = to_tensor(self.atom_types)
        self.bond_types = to_tensor(self.bond_types)
        self.charge_types = to_tensor(self.charge_types)
        self.bond_angles = to_tensor(self.bond_angles)
        self.is_aromatic = to_tensor(self.is_aromatic)
        self.is_in_ring = to_tensor(self.is_in_ring)
        self.hybridization = to_tensor(self.hybridization)
        self.n_atoms_cov = to_tensor(self.n_atoms_cov)
        self.numHs = to_tensor(self.numHs)


def to_tensor(x):
    if isinstance(x, np.ndarray | list):
        return torch.tensor(x)
    else:
        return x


class DatasetInfo:
    def __init__(
        self,
        name: str,
        statistics: Dict[str, Statistics],
        ligand_pocket_histogram: Optional[Union[List[List], Tensor]] = None,
    ):
        super().__init__()

        self.name = name

        self.atom_encoder = ATOM_ENCODER
        self.atom_decoder = ATOM_DECODER
        self.statistics = statistics

        train_n_nodes = statistics["train"].num_nodes
        val_n_nodes = statistics["val"].num_nodes
        test_n_nodes = statistics["test"].num_nodes
        max_n_nodes = max(
            max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys())
        )
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value

        self.n_nodes = n_nodes / n_nodes.sum()
        self.atom_types = to_tensor(statistics["train"].atom_types)
        self.edge_types = to_tensor(statistics["train"].bond_types)
        self.charges_types = to_tensor(statistics["train"].charge_types)
        self.charges_marginals = (self.charges_types * self.atom_types[:, None]).sum(
            dim=0
        )
        self.valency_distribution = to_tensor(statistics["train"].valencies)
        self.max_n_nodes = len(n_nodes) - 1
        # ligand
        self.nodes_dist = DistributionNodes(n_nodes)
        if ligand_pocket_histogram is not None:
            # ligand, pocket
            cds = ConditionalDistributionNodes(histogram=ligand_pocket_histogram)
        else:
            cds = None

        self.conditional_size_distribution: Optional[ConditionalDistributionNodes] = cds

        if hasattr(statistics["train"], "is_aromatic"):
            self.is_aromatic = to_tensor(statistics["train"].is_aromatic)
        if hasattr(statistics["train"], "is_in_ring"):
            self.is_in_ring = to_tensor(statistics["train"].is_in_ring)
        if hasattr(statistics["train"], "hybridization"):
            self.hybridization = to_tensor(statistics["train"].hybridization)
        if hasattr(statistics["train"], "numHs"):
            self.numHs = to_tensor(statistics["train"].numHs)
        if hasattr(statistics["train"], "is_h_donor"):
            self.is_h_donor = to_tensor(statistics["train"].is_h_donor)
        if hasattr(statistics["train"], "is_h_acceptor"):
            self.is_h_acceptor = to_tensor(statistics["train"].is_h_acceptor)

        # raw input charges start from -2
        self.charge_offset = 2
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        self.num_charge_classes = len(self.charges_marginals)

    def one_hot_charges(self, C):
        return F.one_hot(
            (C + self.charge_offset).long(), num_classes=self.num_charge_classes
        ).float()

    def to_dict(self):
        return vars(self)


def load_dataset_info(
    name: str,
    statistics_dict_path: str,
    ligand_pocket_histogram_path: Optional[str] = None,
    dataset: str = "crossdocked",
):

    statistics = load_pickle(statistics_dict_path)
    if dataset == "enamine":
        statistics = {"train": statistics, "valid": statistics, "test": statistics}
    if ligand_pocket_histogram_path is not None:
        lp_histogram = np.load(ligand_pocket_histogram_path)
    else:
        if "ligand_pocket_histogram" not in statistics["train"]:
            lp_histogram = None
        else:
            lp_histogram = statistics["train"]["ligand_pocket_histogram"]
    train = Statistics(**statistics["train"])
    valid = Statistics(**statistics["valid"])
    test = Statistics(**statistics["test"])
    statistics = {"train": train, "val": valid, "test": test}
    dataset_info = DatasetInfo(
        name=name, statistics=statistics, ligand_pocket_histogram=lp_histogram
    )
    return dataset_info

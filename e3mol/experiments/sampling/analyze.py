import logging
from collections import Counter
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.DataStructs import BulkTanimotoSimilarity
from torchmetrics import MaxMetric, MeanMetric

from e3mol.experiments.data.datainfo import DatasetInfo
from e3mol.experiments.data.molecule import Molecule
from e3mol.experiments.sampling.utils import (
    angle_distance,
    atom_types_distance,
    bond_length_distance,
    bond_types_distance,
    canonicalize_list,
    charge_distance,
    check_stability,
    dihedral_distance,
    get_fingerprints_from_smiles_list,
    number_nodes_distance,
    valency_distance,
)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
logging.getLogger("openbabel").setLevel(logging.CRITICAL)


class BasicMolecularMetrics:
    def __init__(self, dataset_info: DatasetInfo, smiles_train=None, device="cpu"):
        self.atom_decoder = (
            dataset_info["atom_decoder"]
            if isinstance(dataset_info, dict)
            else dataset_info.atom_decoder
        )
        self.atom_encoder = (
            dataset_info["atom_encoder"]
            if isinstance(dataset_info, dict)
            else dataset_info.atom_encoder
        )
        self.dataset_info = dataset_info

        self.number_samples = 0
        self.train_smiles, _ = canonicalize_list(smiles_train)
        self.train_fps = get_fingerprints_from_smiles_list(self.train_smiles)

        self.atom_stable = MeanMetric().to(device)
        self.mol_stable = MeanMetric().to(device)

        # Retrieve dataset smiles.
        self.validity_metric = MeanMetric().to(device)
        self.uniqueness = MeanMetric().to(device)
        self.novelty = MeanMetric().to(device)
        self.mean_components = MeanMetric().to(device)
        self.max_components = MaxMetric().to(device)
        self.num_nodes_w1 = MeanMetric().to(device)
        self.atom_types_tv = MeanMetric().to(device)
        self.edge_types_tv = MeanMetric().to(device)
        self.charge_w1 = MeanMetric().to(device)
        self.valency_w1 = MeanMetric().to(device)
        self.bond_lengths_w1 = MeanMetric().to(device)
        self.angles_w1 = MeanMetric().to(device)
        self.dihedrals_w1 = MeanMetric().to(device)

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,
            self.uniqueness,
            self.novelty,
            self.mean_components,
            self.max_components,
            self.num_nodes_w1,
            self.atom_types_tv,
            self.edge_types_tv,
            self.charge_w1,
            self.valency_w1,
            self.bond_lengths_w1,
            self.angles_w1,
            self.dihedrals_w1,
        ]:
            metric.reset()

    def compute_validity(self, generated: List[Molecule], local_rank: int = 0):
        valid_smiles = []
        valid_ids = []
        valid_molecules = []
        num_components = []
        error_message = Counter()
        for i, mol in enumerate(generated):
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=False
                    )
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    else:
                        largest_mol = max(
                            mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms()
                        )
                        Chem.SanitizeMol(largest_mol)
                        smiles = Chem.MolToSmiles(largest_mol)
                        valid_molecules.append(generated[i])
                        valid_smiles.append(smiles)
                        valid_ids.append(i)
                        error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                except Chem.rdchem.AtomKekulizeException or ValueError:  # noqa: B030
                    error_message[3] += 1
        if local_rank == 0:
            print(
                f"Error messages: AtomValence {error_message[1]}, \
                Kekulize {error_message[2]}, Other {error_message[3]}, \
                Disconnected {error_message[4]},"
                f"-- No error {error_message[-1]}"
            )
        self.validity_metric.update(
            value=len(valid_smiles) / len(generated), weight=len(generated)
        )
        num_components = torch.tensor(
            num_components, device=self.mean_components.device
        )
        self.mean_components.update(num_components)
        self.max_components.update(num_components)
        not_connected = 100.0 * error_message[4] / len(generated)
        connected_components = 100.0 - not_connected

        valid_smiles, duplicate_ids = canonicalize_list(valid_smiles)
        valid_molecules = [
            mol for i, mol in enumerate(valid_molecules) if i not in duplicate_ids
        ]

        return (
            valid_smiles,
            valid_molecules,
            connected_components,
            duplicate_ids,
            error_message,
        )

    def compute_sanitize_validity(self, generated: List[Molecule]) -> float:
        if len(generated) < 1:
            return -1.0

        valid = []
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    Chem.SanitizeMol(rdmol)
                except ValueError:
                    continue

                valid.append(rdmol)

        return len(valid) / len(generated)

    def compute_uniqueness(self, valid: List[str]):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique: List[str]):
        num_novel = 0
        novel = []
        if self.train_smiles is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.train_smiles:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated: List[Molecule], local_rank: int = 0):
        # Validity
        (
            valid_smiles,
            valid_molecules,
            connected_components,
            duplicates,
            error_message,
        ) = self.compute_validity(generated, local_rank=local_rank)

        validity = self.validity_metric.compute()
        uniqueness, novelty = 0, 0
        mean_components = self.mean_components.compute()
        max_components = self.max_components.compute()

        # Uniqueness
        if len(valid_smiles) > 0:
            self.uniqueness.update(
                value=1 - (len(duplicates) / len(valid_smiles)),
                weight=len(valid_smiles),
            )
            uniqueness = self.uniqueness.compute()

            if self.train_smiles is not None:
                novel = []
                for smiles in valid_smiles:
                    if smiles not in self.train_smiles:
                        novel.append(smiles)
                self.novelty.update(
                    value=len(novel) / len(valid_smiles), weight=len(valid_smiles)
                )
            novelty = self.novelty.compute()

        num_molecules = int(self.validity_metric.weight.item())
        if local_rank == 0:
            print(
                f"Validity over {num_molecules} molecules:" f" {validity * 100 :.2f}%"
            )
            print(
                f"Number of connected components of {num_molecules} molecules: "
                f"mean:{mean_components:.2f} max:{max_components:.2f}"
            )
            print(
                f"Connected components of {num_molecules} molecules: "
                f"{connected_components:.2f}"
            )
            print(f"Uniqueness over {num_molecules} molecules: " f"{uniqueness:.2f}")
            print(f"Novelty over {num_molecules} molecules: " f"{novelty:.2f}")

        return (
            valid_smiles,
            valid_molecules,
            validity,
            novelty,
            uniqueness,
            connected_components,
        )

    def compute_statistics(
        self, generated: List[Molecule], local_rank: int = 0, test: bool = False
    ):
        # Compute statistics
        stat = (
            self.dataset_info.statistics["test"]
            if test
            else self.dataset_info.statistics["val"]
        )

        self.num_nodes_w1(number_nodes_distance(generated, stat.num_nodes))

        atom_types_tv, _ = atom_types_distance(
            generated, stat.atom_types, save_histogram=test
        )
        self.atom_types_tv(atom_types_tv)

        edge_types_tv, _, sparsity_level = bond_types_distance(
            generated, stat.bond_types, save_histogram=test
        )
        print(
            f"Sparsity level on local rank {local_rank}: {int(100 * sparsity_level)} %"
        )
        self.edge_types_tv(edge_types_tv)
        charge_w1, _ = charge_distance(
            generated, stat.charge_types, stat.atom_types, self.dataset_info
        )
        self.charge_w1(charge_w1)
        valency_w1, _ = valency_distance(
            generated, stat.valencies, stat.atom_types, self.atom_encoder
        )
        self.valency_w1(valency_w1)
        bond_lengths_w1, _ = bond_length_distance(
            generated, stat.bond_lengths, stat.bond_types
        )
        self.bond_lengths_w1(bond_lengths_w1)
        if sparsity_level < 0.7:
            if local_rank == 0:
                print("Too many edges, skipping angle distance computation.")
            angles_w1 = 0
            _ = [-1] * len(self.atom_decoder)
        else:
            angles_w1, _ = angle_distance(
                generated,
                stat.bond_angles,
                stat.atom_types,
                stat.valencies,
                atom_decoder=self.atom_decoder,
                save_histogram=test,
            )
        self.angles_w1(angles_w1)
        dihedrals_w1, _ = dihedral_distance(
            generated, stat.dihedrals, stat.bond_types, save_histogram=test
        )
        self.dihedrals_w1(dihedrals_w1)
        statistics_log = {
            "sampling/NumNodesW1": self.num_nodes_w1.compute().item(),
            "sampling/AtomTypesTV": self.atom_types_tv.compute().item(),
            "sampling/EdgeTypesTV": self.edge_types_tv.compute().item(),
            "sampling/ChargeW1": self.charge_w1.compute().item(),
            "sampling/ValencyW1": self.valency_w1.compute().item(),
            "sampling/BondLengthsW1": self.bond_lengths_w1.compute().item(),
            "sampling/AnglesW1": self.angles_w1.compute().item(),
            "sampling/DihedralsW1": self.dihedrals_w1.compute().item(),
        }

        return statistics_log

    def get_bulk_similarity_with_train(self, generated_smiles):
        fps = get_fingerprints_from_smiles_list(generated_smiles)
        scores = []

        for fp in fps:
            scores.append(BulkTanimotoSimilarity(fp, self.train_fps))
        return np.mean(scores)

    def get_bulk_diversity(self, generated_smiles):
        fps = get_fingerprints_from_smiles_list(generated_smiles)
        scores = []
        for i, fp in enumerate(fps):
            fps_tmp = fps.copy()
            del fps_tmp[i]
            scores.append(BulkTanimotoSimilarity(fp, fps_tmp))
        return 1 - np.mean(scores)

    def __call__(
        self,
        molecules: List[Molecule],
        local_rank: int = 0,
        remove_hs: bool = False,
        filter_by_posebusters: bool = False,
        filter_by_lipinski: bool = False,
        pdb_file: Optional[str] = None,
        test: bool = False,
    ):
        stable_molecules = []

        if not remove_hs:
            # Atom and molecule stability only performend with explicit hydrogens
            if local_rank == 0:
                print("Analyzing molecule stability")
            for mol in molecules:
                mol_stable, at_stable, num_bonds = check_stability(
                    mol, self.atom_decoder
                )
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)
                if mol_stable:
                    stable_molecules.append(mol)
            stability_dict = {
                "mol_stable": self.mol_stable.compute().item(),
                "atm_stable": self.atom_stable.compute().item(),
            }
        else:
            stability_dict = {}
            if local_rank == 0:
                print("No explicit hydrogens - skipping molecule stability metric")

        # Validity, uniqueness, novelty
        (
            valid_smiles,
            valid_molecules,
            validity,
            novelty,
            uniqueness,
            connected_components,
        ) = self.evaluate(molecules, local_rank=local_rank)

        if filter_by_posebusters:
            raise NotImplementedError("PoseBusters filtering not implemented yet.")
        if filter_by_lipinski:
            raise NotImplementedError("Lipinski filtering not implemented yet.")

        sanitize_validity = self.compute_sanitize_validity(molecules)
        novelty = novelty if isinstance(novelty, int) else novelty.item()
        uniqueness = uniqueness if isinstance(uniqueness, int) else uniqueness.item()
        validity_dict = {
            "validity": validity.item(),
            "sanitize_validity": sanitize_validity,
            "novelty": novelty,
            "uniqueness": uniqueness,
        }

        # statistics
        try:
            statistics_dict = self.compute_statistics(
                molecules, local_rank=local_rank, test=test
            )
        except Exception as e:
            print(f"Error computing statistics: {e}")
            statistics_dict = {}
        statistics_dict["connected_components"] = connected_components
        self.number_samples = len(valid_smiles)
        similarity = self.get_bulk_similarity_with_train(valid_smiles)
        diversity = self.get_bulk_diversity(valid_smiles)
        statistics_dict["bulk_similarity_with_train"] = similarity
        statistics_dict["bulk_diversity_generated_set"] = diversity

        self.reset()

        outs = {
            "stability_dict": stability_dict,
            "validity_dict": validity_dict,
            "statistics_dict": statistics_dict,
            "valid_molecules": valid_molecules,
            "stable_molecules": stable_molecules,
            "valid_smiles": valid_smiles,
        }

        return outs


def analyze_stability_for_molecules(
    dataset_info: DatasetInfo,
    device: str,
    smiles_train: List[str],
    molecule_list: List[Molecule],
    remove_hs: bool = False,
    local_rank: int = 0,
    test: bool = False,
    filter_by_posebusters: bool = False,
    filter_by_lipinski: bool = False,
    pdb_file: Optional[str] = None,
):
    metrics = BasicMolecularMetrics(
        dataset_info=dataset_info, smiles_train=smiles_train, device=device
    )
    outs = metrics(
        molecule_list,
        local_rank=local_rank,
        remove_hs=remove_hs,
        filter_by_posebusters=filter_by_posebusters,
        filter_by_lipinski=filter_by_lipinski,
        pdb_file=pdb_file,
        test=test,
    )
    return outs

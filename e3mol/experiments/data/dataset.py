from itertools import zip_longest
from typing import Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import Chem, RDLogger
from scipy.ndimage import gaussian_filter
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from e3mol.experiments.data.datainfo import ATOM_ENCODER, CHARGES_DICT
from e3mol.experiments.data.pl_utils import mol_to_torch_geometric
from e3mol.experiments.data.stats import compute_all_statistics
from e3mol.experiments.data.utils import load_pickle, save_pickle

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]


def get_empirical_distribution_n_lig_pocket(lig_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1, np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(
        f"Original histogram: {np.count_nonzero(joint_histogram)}/"
        f"{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled"
    )

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram,
            sigma=smooth_sigma,
            order=0,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )

        print(
            f"Smoothed histogram: {np.count_nonzero(filtered_histogram)}/"
            f"{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled"
        )

        joint_histogram = filtered_histogram

    return joint_histogram


class LigandPocketDataset(InMemoryDataset):
    def __init__(
        self,
        split: str,
        root: str,
        remove_hs=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        center_of_gravity=False,
        compute_bond_distance_angles=True,
        kekulize=True,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.remove_hs = remove_hs
        self.center_of_gravity = center_of_gravity
        self.compute_bond_distance_angles = compute_bond_distance_angles
        self.atom_encoder = ATOM_ENCODER
        self.kekulize = kekulize

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistics = load_pickle(self.processed_paths[1])
        self.smiles = load_pickle(self.processed_paths[2])

    @property
    def raw_file_names(self):
        if self.split == "train":
            return ["train.npz"]
        elif self.split == "val":
            return ["val.npz"]
        else:
            return ["test.npz"]

    def processed_file_names(self):
        h = "noh" if self.remove_hs else "h"
        if self.split == "train":
            return [
                f"train_{h}.pt",
                f"train_stats_dict_{h}.pickle",
                f"train_smiles_{h}.pickle",
            ]
        elif self.split == "val":
            return [
                f"val_{h}.pt",
                f"val_stats_dict_{h}.pickle",
                f"val_smiles_{h}.pickle",
            ]
        else:
            return [
                f"test_{h}.pt",
                f"test_stats_dict_{h}.pickle",
                f"test_smiles_{h}.pickle",
            ]

    def download(self):
        raise ValueError(
            "Download and preprocessing is manual.\
            If the data is already downloaded, "
            f"check that the paths are correct. \
            Root dir = {self.root} -- raw files {self.raw_paths}"
        )

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        data_list_lig = []
        all_smiles = []

        with np.load(self.raw_paths[0], allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        if self.split == "train":
            smooth_sigma = 1.0
            lig_mask = data["lig_mask"]
            pocket_mask = data["pocket_mask"]
            ligand_pocket_histogram = get_empirical_distribution_n_lig_pocket(
                lig_mask, pocket_mask, smooth_sigma=smooth_sigma
            )
            del pocket_mask, lig_mask
        else:
            ligand_pocket_histogram = None

        # split data based on mask
        mol_data = {}

        for k, v in data.items():

            if k == "names" or k == "receptors" or k == "lig_mol" or k == "lig_smiles":
                mol_data[k] = v
                continue

            sections = (
                np.where(np.diff(data["lig_mask"]))[0] + 1
                if "lig" in k
                else np.where(np.diff(data["pocket_mask"]))[0] + 1
            )
            if k == "lig_atom" or k == "pocket_atom":
                mol_data[k] = [
                    torch.tensor([ATOM_ENCODER[a] for a in atoms])
                    for atoms in np.split(v, sections)
                ]
            elif k in [
                "pocket_resids",
                "pocket_chainids",
                "pocket_resnames",
                "pocket_atom_names",
            ]:
                mol_data[k] = [ids for ids in np.split(v, sections)]
            else:
                if k == "pocket_one_hot":
                    pocket_one_hot_mask = data["pocket_mask"][data["pocket_ca_mask"]]
                    pocket_one_hot = data["pocket_one_hot"]
                    sections = np.where(np.diff(pocket_one_hot_mask))[0] + 1
                    mol_data["pocket_one_hot_mask"] = [
                        torch.from_numpy(x)
                        for x in np.split(pocket_one_hot_mask, sections)
                    ]

                    sections = np.where(np.diff(data["pocket_mask"]))[0] + 1
                    mol_data["pocket_one_hot"] = [
                        torch.from_numpy(x) for x in np.split(pocket_one_hot, sections)
                    ]
                else:
                    mol_data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == "lig_mask":
                mol_data["num_lig_atoms"] = torch.tensor(
                    [len(x) for x in mol_data["lig_mask"]]
                )
            elif k == "pocket_mask":
                mol_data["num_pocket_nodes"] = torch.tensor(
                    [len(x) for x in mol_data["pocket_mask"]]
                )
            elif k == "pocket_one_hot":
                mol_data["num_resids_nodes"] = torch.tensor(
                    [len(x) for x in mol_data["pocket_one_hot_mask"]]
                )
        for i, (
            mol_lig,
            coords_lig,
            atoms_lig,
            mask_lig,
            coords_pocket,
            atoms_pocket,
            mask_pocket,
            pocket_ca_mask,
            name,
            pocket_one_hot,
        ) in enumerate(
            tqdm(
                zip_longest(
                    mol_data["lig_mol"],
                    mol_data["lig_coords"],
                    mol_data["lig_atom"],
                    mol_data["lig_mask"],
                    mol_data["pocket_coords"],
                    mol_data["pocket_atom"],
                    mol_data["pocket_mask"],
                    mol_data["pocket_ca_mask"],
                    mol_data["names"],
                    mol_data["pocket_one_hot"],
                    fillvalue=None,
                ),
                total=len(mol_data["lig_mol"]),
            )
        ):
            try:
                # atom_types = [atom_decoder[int(a)] for a in atoms_lig]
                # smiles_lig, conformer_lig = get_mol_babel(coords_lig, atom_types)
                smiles_lig = Chem.MolToSmiles(mol_lig)
                data = mol_to_torch_geometric(
                    mol_lig,
                    ATOM_ENCODER,
                    smiles_lig,
                    remove_hydrogens=self.remove_hs,
                    cog_proj=False,
                    kekulize=self.kekulize,
                )
            except Exception as e:
                print(e)
                print(f"Ligand {i} failed")
                continue
            data.pos_lig = coords_lig
            data.x_lig = atoms_lig
            data.pos_pocket = coords_pocket
            data.x_pocket = atoms_pocket
            data.lig_mask = mask_lig
            data.pocket_mask = mask_pocket
            data.pocket_ca_mask = pocket_ca_mask
            data.pocket_name = name
            data.pocket_one_hot = pocket_one_hot
            # getting the sub-ids
            # data.sub_ids = load_sub_ids(ligand_name) [[...], [...], [...]]

            all_smiles.append(smiles_lig)
            data_list_lig.append(data)

        if self.center_of_gravity:
            for i in range(len(data_list_lig)):
                mean = (
                    data_list_lig[i].pos.sum(0) + data_list_lig[i].pos_pocket.sum(0)
                ) / (len(data_list_lig[i].pos) + len(data_list_lig[i].pos_pocket))
                data_list_lig[i].pos = data_list_lig[i].pos - mean
                data_list_lig[i].pos_pocket = data_list_lig[i].pos_pocket - mean

        torch.save(self.collate(data_list_lig), self.processed_paths[0])

        print(f"Finished processing. Saved at {self.processed_paths[0]}")
        print("Calculating statistics")

        statistics = compute_all_statistics(
            data_list_lig,
            self.atom_encoder,
            charges_dic=CHARGES_DICT,
            additional_feats=True,
            normalize=True,
            bond_angles_distribution=self.compute_bond_distance_angles,
        )
        statistics = statistics.to_dict()
        statistics["ligand_pocket_histogram"] = ligand_pocket_histogram
        save_pickle(statistics, self.processed_paths[1])
        save_pickle(set(all_smiles), self.processed_paths[2])

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ):
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)


class LigandPocketDataModule(pl.LightningDataModule):
    def __init__(self, cfg, transform=None, validation_transform=None):
        super().__init__()
        self.datadir = cfg.dataset_root
        self.pin_memory = True
        self.test_batch_size = 1
        self.cfg = cfg
        self.transform = transform
        self.validation_transform = validation_transform

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = LigandPocketDataset(
                split="train",
                root=self.datadir,
                remove_hs=self.cfg.remove_hs,
                center_of_gravity=False,
                compute_bond_distance_angles=True,
                transform=self.transform,
            )
            self.val_dataset = LigandPocketDataset(
                split="val",
                root=self.datadir,
                remove_hs=self.cfg.remove_hs,
                center_of_gravity=False,
                compute_bond_distance_angles=True,
                transform=self.validation_transform,
            )
            self.test_dataset = LigandPocketDataset(
                split="test",
                root=self.datadir,
                remove_hs=self.cfg.remove_hs,
                center_of_gravity=False,
                compute_bond_distance_angles=True,
                transform=self.validation_transform,
            )

            # save joined statistics for train/val/test as dictionary
            statistics = {
                "train": self.train_dataset.statistics,
                "valid": self.val_dataset.statistics,
                "test": self.test_dataset.statistics,
            }
            save_path = self.datadir + "/processed/all_stats_dict_noh.pickle"
            save_pickle(statistics, save_path, exist_ok=False)

    def get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.cfg.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.cfg.batch_size
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            follow_batch=["pos", "pos_pocket"],
        )

        return dl

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset, "test")

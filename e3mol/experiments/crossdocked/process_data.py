import argparse
import random
import shutil
from pathlib import Path
from time import time

import numpy as np
import torch
from tqdm import tqdm

from e3mol.experiments.data.process_protein_ligand_data import (
    PDBParser,
    process_protein_ligand,
    save_all_as_np,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-H", action="store_true")
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--dist-cutoff", type=float, default=8.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--residue-com", default=False, action="store_true")

    args = parser.parse_args()

    datadir = args.base_dir / "crossdocked_pocket10/"
    # Make output directory
    if args.out_dir is None:
        suffix = "_crossdock_noH" if args.no_H else "_crossdock_H"
        suffix += "_ca_only_temp" if args.ca_only else "_full_temp"
        suffix += f"_cutoff{args.dist_cutoff}"
        processed_dir = Path(args.base_dir, f"processed{suffix}")
    else:
        processed_dir = args.out_dir

    processed_dir.mkdir(exist_ok=True, parents=True)

    # Read data split
    split_path = Path(args.base_dir, "split_by_name.pt")
    data_split = torch.load(split_path)

    # There is no validation set, copy 300 training examples (the validation set
    # is not very important in this application)
    # Note: before we had a data leak but it should not matter too much as most
    # metrics monitored during training are independent of the pockets
    random.seed(args.random_seed)
    data_split["val"] = random.sample(data_split["train"], 300)

    n_train_before = len(data_split["train"])
    n_val_before = len(data_split["val"])
    n_test_before = len(data_split["test"])

    failed_save = []

    n_samples_after = {}

    for split in data_split.keys():

        lig_coords = []
        lig_atom = []
        lig_mask = []
        lig_mol = []
        docking_scores = []
        pocket_coords = []
        pocket_atom = []
        pocket_atom_names = []
        pocket_mask = []
        pocket_resids = []
        pocket_chainids = []
        pocket_resnames = []
        pocket_one_hot_resids = []
        pocket_ca_mask = []
        pdb_and_mol_ids = []
        count_protein = []
        count_ligand = []
        count_total = []
        count = 0

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        tic = time()
        num_failed = 0
        pbar = tqdm(data_split[split])
        pbar.set_description(f"#failed: {num_failed}")

        for pocket_fn, ligand_fn in pbar:

            sdffile = datadir / f"{ligand_fn}"
            pdbfile = datadir / f"{pocket_fn}"

            try:
                _ = PDBParser(QUIET=True).get_structure("", pdbfile)
            except Exception:
                num_failed += 1
                failed_save.append((pocket_fn, ligand_fn))
                print(failed_save[-1])
                pbar.set_description(f"#failed: {num_failed}")
                continue
            try:
                ligand_data, pocket_data = process_protein_ligand(
                    pdbfile,
                    sdffile,
                    dist_cutoff=args.dist_cutoff,
                    ca_only=args.ca_only,
                    no_H=args.no_H,
                    residue_com=args.residue_com,
                )
            except (
                KeyError,
                AssertionError,
                FileNotFoundError,
                IndexError,
                ValueError,
            ) as e:
                print(type(e).__name__, e, pocket_fn, ligand_fn)
                num_failed += 1
                pbar.set_description(f"#failed: {num_failed}")
                continue

            pocket_name = ("-").join(pocket_fn.split("/")[1].split("_"))
            ligand_name = ("-").join(ligand_fn.split("/")[1].split("_"))
            pdb_and_mol_ids.append(f"{pocket_name}_{ligand_name}")
            lig_coords.append(ligand_data["lig_coords"])
            lig_mask.append(count * np.ones(len(ligand_data["lig_coords"])))
            lig_atom.append(ligand_data["lig_atoms"])
            lig_mol.append(ligand_data["lig_mol"])
            pocket_coords.append(pocket_data["pocket_coords"])
            pocket_atom.append(pocket_data["pocket_atoms"])
            pocket_atom_names.append(pocket_data["pocket_atom_names"])
            pocket_mask.append(count * np.ones(len(pocket_data["pocket_coords"])))
            pocket_resids.append(pocket_data["pocket_resids"])
            pocket_chainids.append(pocket_data["pocket_chainids"])
            pocket_resnames.append(pocket_data["pocket_resnames"])
            # new
            if not args.ca_only:
                pocket_one_hot_resids.append(pocket_data["pocket_one_hot"])
                pocket_ca_mask.append(pocket_data["pocket_ca_mask"])

            count_protein.append(pocket_data["pocket_coords"].shape[0])
            count_ligand.append(ligand_data["lig_coords"].shape[0])
            count_total.append(
                pocket_data["pocket_coords"].shape[0]
                + ligand_data["lig_coords"].shape[0]
            )
            count += 1

            # if split in {"val", "test"}:
            # Copy PDB file
            new_rec_name = Path(pdbfile).stem.replace("_", "-")
            pdb_file_out = Path(pdb_sdf_dir, f"{new_rec_name}.pdb")
            shutil.copy(pdbfile, pdb_file_out)

            # Copy SDF file
            new_lig_name = new_rec_name + "_" + Path(sdffile).stem.replace("_", "-")
            sdf_file_out = Path(pdb_sdf_dir, f"{new_lig_name}.sdf")
            shutil.copy(sdffile, sdf_file_out)

            # specify pocket residues
            with open(Path(pdb_sdf_dir, f"{new_lig_name}_pocket_ids.txt"), "w") as f:
                f.write(" ".join(pocket_data["pocket_chainids"]))

        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_atom = np.concatenate(lig_atom, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        lig_mol = np.array(lig_mol)
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_atom = np.concatenate(pocket_atom, axis=0)
        pocket_atom_names = np.concatenate(pocket_atom_names, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)
        pocket_resids = np.concatenate(pocket_resids, axis=0)
        pocket_chainids = np.concatenate(pocket_chainids, axis=0)
        pocket_resnames = np.concatenate(pocket_resnames, axis=0)
        docking_scores = np.array(docking_scores)

        if not args.ca_only:
            pocket_one_hot_resids = np.concatenate(pocket_one_hot_resids, axis=0)
            pocket_ca_mask = np.concatenate(pocket_ca_mask, axis=0)
        else:
            pocket_one_hot_resids = np.array([])
            pocket_ca_mask = np.array([])

        proc_raw = processed_dir / "raw"
        proc_raw.mkdir(exist_ok=True)

        save_all_as_np(
            proc_raw / f"{split}.npz",
            pdb_and_mol_ids,
            lig_coords,
            lig_atom,
            lig_mask,
            lig_mol,
            pocket_coords,
            pocket_atom,
            pocket_atom_names,
            pocket_mask,
            pocket_resids,
            pocket_chainids,
            pocket_resnames,
            pocket_one_hot=pocket_one_hot_resids,
            pocket_ca_mask=pocket_ca_mask,
        )

        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic)/60.0:.2f} minutes")
        print("Failed to process:", failed_save)

    # Create summary string
    summary_string = "# SUMMARY\n\n"
    summary_string += "# Before processing\n"
    summary_string += f"num_samples train: {n_train_before}\n"
    summary_string += f"num_samples val: {n_val_before}\n"
    summary_string += f"num_samples test: {n_test_before}\n\n"
    summary_string += "# After processing\n"
    summary_string += f"num_samples train: {n_samples_after['train']}\n"
    summary_string += f"num_samples val: {n_samples_after['val']}\n"
    summary_string += f"num_samples test: {n_samples_after['test']}\n\n"
    print(summary_string)


if __name__ == "__main__":
    main()

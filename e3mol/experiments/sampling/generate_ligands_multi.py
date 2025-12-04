import argparse
import itertools as it
import json
import multiprocessing as mp
import os
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import PandasTools

from e3mol.experiments.data.datainfo import load_dataset_info
from e3mol.experiments.data.dataset import LigandPocketDataModule as DataModule
from e3mol.experiments.inference.utils import create_batch_from_pl_files
from e3mol.experiments.sampling.analyze import analyze_stability_for_molecules
from e3mol.experiments.sampling.evaluate_pose import wrap_pose_evaluation_mp
from e3mol.experiments.sampling.ring_analysis import get_ring_info
from e3mol.experiments.sampling.utils import split_list
from e3mol.experiments.trainer import Trainer as TrainerDiffusion
from e3mol.experiments.trainer import TrainerFlow


def check_for_disconnected_components(rdmol):
    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
    return len(mol_frags) < 2


def wrap_is_valid(mol):
    try:
        Chem.SanitizeMol(mol)
        return check_for_disconnected_components(mol)
    except Exception:
        return False


def load_model_dataset_info(
    model_path: str, dataset_root: str, model_type: str = "diffusion"
):
    hparams = torch.load(model_path, map_location="cpu")["hyper_parameters"]
    hparams = namedtuple("hparams", hparams.keys())(*hparams.values())
    hparams = hparams._replace(dataset_root=dataset_root)
    if model_type == "flow":
        from e3mol.experiments.trainer import TrainerFlow as Trainer
    elif model_type == "diffusion":
        from e3mol.experiments.trainer import Trainer as Trainer
    else:
        raise ValueError

    datamodule = DataModule(hparams, transform=None)
    datamodule.setup()
    statistics_dict_path = hparams.dataset_root + "/processed/all_stats_dict_noh.pickle"
    dataset_info = load_dataset_info(
        name=hparams.dataset,
        statistics_dict_path=statistics_dict_path,
        ligand_pocket_histogram_path=None,
    )

    device = torch.device("cuda")
    model = Trainer.load_from_checkpoint(
        model_path,
        dataset_info=dataset_info,
        pocket_noise_std=hparams.pocket_noise_std,
        ckpt_path=None,
        smiles=datamodule.train_dataset.smiles,
    )
    model.smiles_train = datamodule.train_dataset.smiles
    model = model.to(device)
    return model


def process_one_sdf_file(
    sdf_file: Path,
    pdb_file: Path,
    model: TrainerDiffusion | TrainerFlow,
    batch_size: int,
    num_integration_steps: int = 100,
    pocket_cutoff: float = 5.0,
    nodes_bias: int = 5,
    vary_n_nodes: bool = True,
    num_ligands_per_pocket_to_sample: int = 100,
    device: str = "cuda",
    prior_n_atoms: str = "reference",
    max_sample_iter: int = 5,
    clash_guidance: bool = False,
    clash_guidance_scale: float = 0.1,
    score_dynamics: bool = False,
    score_scale: float = 0.5,
    noise_dynamics: bool = False,
    noise_scale: Optional[float] = None,
    dt_pow=1.0,
    noise_schedule: str = "sine",
    discrete_gat: bool = False,
):

    start = datetime.now()
    tmp_molecules = []
    k = 0
    iters = num_ligands_per_pocket_to_sample // batch_size + 1
    iters = iters + max_sample_iter

    for k in range(iters):
        data = create_batch_from_pl_files(
            pdb_file=pdb_file,
            sdf_file=sdf_file,
            batch_size=batch_size,
            pocket_cutoff=pocket_cutoff,
            removeHs_ligand_post=True,
            removeHs_pocket_post=True,
        )
        data = data.to(device)
        num_graphs = len(data.batch.bincount())
        if prior_n_atoms == "reference":
            num_nodes_lig = data.batch.bincount().to(device)
        elif prior_n_atoms == "conditional":

            if model.hparams.model_type == "protein-ligand":
                num_nodes_lig = model.dataset_info.conditional_size_distribution.sample_conditional(  # noqa: B950
                    n1=None, n2=data.pos_pocket_batch.bincount()
                ).to(
                    device
                )
            else:
                num_nodes_lig = model.dataset_info.nodes_dist.sample_n(
                    n_samples=num_graphs, device=device
                ).to(device)
        else:
            raise ValueError("Invalid prior_n_atoms")

        if vary_n_nodes:
            nodes_bias_ = torch.randint(
                low=-nodes_bias, high=nodes_bias + 1, size=(num_graphs,), device=device
            )
        else:
            nodes_bias_ = nodes_bias

        num_nodes_lig = num_nodes_lig + nodes_bias_

        with torch.no_grad():
            molecules, _, _ = model.model.reverse_sampling(
                N=num_integration_steps,
                num_graphs=num_graphs,
                device=device,
                num_nodes_lig=num_nodes_lig,
                pocket_data=data,
                clash_guidance=clash_guidance,
                clash_guidance_scale=clash_guidance_scale,
                cat_noise=True,
                score_dynamics=score_dynamics,
                score_scale=score_scale,
                pos_noise=noise_dynamics,
                dt_pow=dt_pow,
                strategy="linear",
                noise_schedule=noise_schedule,
                discrete_gat=discrete_gat,
            )

        del data
        torch.cuda.empty_cache()
        tmp_molecules.extend(molecules)
        is_valid = [wrap_is_valid(m.rdkit_mol) for m in tmp_molecules]
        validity = np.mean(is_valid)
        print(
            f"Generation step = {k}, \
            Total Sampled = {len(tmp_molecules)}, \
            Validity current set = {validity:.4f}"
        )
        if sum(is_valid) > num_ligands_per_pocket_to_sample:
            print(
                f"Stopping generation at step {k} \
                because {sum(is_valid)} valid molecules have been sampled"
            )
            break

    # evaluate
    outs = analyze_stability_for_molecules(
        dataset_info=model.dataset_info,
        device="cpu",
        smiles_train=model.smiles_train,
        molecule_list=tmp_molecules,
        remove_hs=model.hparams.remove_hs,
        local_rank=0,
        test=True,  # False
        filter_by_lipinski=False,
        filter_by_posebusters=False,
        pdb_file=None,
    )

    valid_and_unique_molecules = outs["valid_molecules"].copy()
    print(
        f"Sampled {len(valid_and_unique_molecules)}\
        molecules in {datetime.now() - start}"
    )
    valid_and_unique_rdmols = [m.rdkit_mol for m in valid_and_unique_molecules]
    if len(valid_and_unique_rdmols) > num_ligands_per_pocket_to_sample:
        print(
            f"Selecting first {num_ligands_per_pocket_to_sample} \
            entries in generated molecue list"
        )
        valid_and_unique_rdmols = valid_and_unique_rdmols[
            :num_ligands_per_pocket_to_sample
        ]

    out = {
        "mols": valid_and_unique_rdmols,
        "validity_dict": outs["validity_dict"],
        "statistics_dict": outs["statistics_dict"],
    }

    return out


def evaluate_samples(
    generated: Path | List[Chem.Mol],
    ref_file: Path,
    pdb_file: Path,
    pdbqt_file: Path,
    num_processes: int,
    qvina2_path: str,
    score_only: bool = True,
    clash_tolerance: float = 0.5,
):

    assert qvina2_path is not None, "qvina2_path must be provided"
    assert os.path.exists(qvina2_path), f"qvina2_path {qvina2_path} does not exist"

    if isinstance(generated, Path | str):
        assert str(generated).endswith(".sdf")
        suppl = Chem.SDMolSupplier(str(generated), sanitize=False, removeHs=False)
        mols = [Chem.AddHs(m, addCoords=True) for m in suppl]
    else:
        mols = generated

    if len(mols) > num_processes:
        mol_chunks = split_list(mols, num_processes)
        print(
            f"Evaluating {len(mols)} generated molecules using {num_processes} processes"
        )
        per_chunk = [len(a) for a in mol_chunks]
        if np.min(per_chunk) == 1:
            mol_chunks = split_list(mols, num_processes - 1)
    else:
        mol_chunks = [mols]
        print(f"Evaluating {len(mols)} generated molecules using 1 process")

    nprocs = len(mol_chunks)
    start = datetime.now()
    iterables = zip(
        mol_chunks,
        it.repeat(ref_file, nprocs),
        it.repeat(pdb_file, nprocs),
        it.repeat(pdbqt_file, nprocs),
        it.repeat(qvina2_path, nprocs),
        it.repeat(score_only, nprocs),
        it.repeat(clash_tolerance, nprocs),
    )
    with mp.Pool(num_processes) as pool:
        outs = pool.starmap(func=wrap_pose_evaluation_mp, iterable=iterables)
    outs_df = pd.concat(outs, axis=0).reset_index(drop=True)
    end = datetime.now() - start
    print(f"Evaluation time: {end}")
    return outs_df


def wrap_one_sdf_file(
    sdf_file: Path,
    model: TrainerDiffusion | TrainerFlow,
    gen_path: Path | str,
    qvina2_path: str,
    num_integration_steps: int = 100,
    batch_size: int = 32,
    pocket_cutoff: float = 5.0,
    nodes_bias: int = 5,
    vary_n_nodes: bool = True,
    max_sample_iter: int = 5,
    num_ligands_per_pocket_to_sample: int = 100,
    prior_n_atoms: str = "reference",
    num_processes: int = 16,
    score_only: bool = True,
    clash_tolerance: float = 0.5,
    clash_guidance: bool = False,
    clash_guidance_scale: float = 0.1,
    score_dynamics: bool = False,
    score_scale: float = 0.5,
    noise_dynamics: bool = False,
    noise_scale: float = 1.0,
    dt_pow: float = 1.0,
    noise_schedule: str = "sine",
    discrete_gat: bool = False,
):

    if "spindr" not in str(sdf_file):
        name = sdf_file.stem.split("_")[0]
        pdbqt_file = sdf_file.parents[0] / f"pdbqt/{name}.pdbqt"
        pdb_file = sdf_file.parents[0] / f"{name}.pdb"
    else:
        name = sdf_file.stem
        pdbqt_file = sdf_file.parents[0] / f"pdbqt/{name}.pdbqt"
        pdb_file = sdf_file.parents[0] / f"{name}.pdb"

    outs = process_one_sdf_file(
        num_integration_steps=num_integration_steps,
        sdf_file=sdf_file,
        pdb_file=pdb_file,
        model=model,
        batch_size=batch_size,
        pocket_cutoff=pocket_cutoff,
        nodes_bias=nodes_bias,
        vary_n_nodes=vary_n_nodes,
        num_ligands_per_pocket_to_sample=num_ligands_per_pocket_to_sample,
        device="cuda",
        prior_n_atoms=prior_n_atoms,
        max_sample_iter=max_sample_iter,
        clash_guidance=clash_guidance,
        clash_guidance_scale=clash_guidance_scale,
        score_dynamics=score_dynamics,
        score_scale=score_scale,
        noise_dynamics=noise_dynamics,
        noise_scale=noise_scale,
        dt_pow=dt_pow,
        noise_schedule=noise_schedule,
        discrete_gat=discrete_gat,
    )

    evals = evaluate_samples(
        generated=outs["mols"],
        ref_file=sdf_file,
        pdb_file=pdb_file,
        pdbqt_file=pdbqt_file,
        num_processes=num_processes,
        qvina2_path=qvina2_path,
        score_only=score_only,
        clash_tolerance=clash_tolerance,
    )
    evals["mol_id"] = np.arange(len(evals))
    gen_file = str(gen_path) + f"/{sdf_file.stem}_gen.sdf"
    gen_csv = str(gen_path) + f"/{sdf_file.stem}_gen.csv"
    print("Writing generated molecules to", gen_file)
    print("Writing reduced metrics to", gen_csv)
    # ring analysis
    ring_info = [get_ring_info(mol) for mol in evals.mols]
    ring_info_df = pd.DataFrame(ring_info)
    evals = pd.concat([evals, ring_info_df], axis=1)
    sel_col = evals.columns.tolist()
    sel_col.remove("mols")
    PandasTools.WriteSDF(
        evals,
        gen_file,
        molColName="mols",
        idName="mol_id",
        properties=sel_col,
        allNumeric=False,
        forceV3000=False,
    )
    evals[sel_col].to_csv(gen_csv, index=False)
    save_dict = {
        "pose_dict": {
            "pb_valid": evals.pb_valid.mean(0),
            "clashes": evals.clashes.mean(0),
        },
        "validity_dict": outs["validity_dict"],
        "statistics_dict": outs["statistics_dict"],
    }
    return save_dict


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--mp-index', default=0, type=int)
    parser.add_argument("--num-gpus", default=8, type=int)
    parser.add_argument('--model-path', type=str,
                        help='Path to trained model')
    parser.add_argument('--dataset-root', type=str,
                        help='Path to the dataset root')
    parser.add_argument('--num-ligands-per-pocket-to-sample',
                        default=100,
                        type=int,
                        help='How many ligands per pocket to sample. Defaults to 100')
    parser.add_argument('--batch-size', default=50, type=int)
    parser.add_argument('--num-processes', default=12, type=int)
    parser.add_argument('--model-type', default="flow", type=str,
                        choices=["flow", "diffusion"])
    parser.add_argument('--dist-cutoff', default=5.0, type=float)
    parser.add_argument('--max-sample-iter', default=5, type=int)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument(
        "--pdbqt-dir",
        type=Path,
        default=None,
        help="Directory where all full protein pdbqt files are stored.\
        If not available, there will be calculated on the fly.",
    )
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--vary-n-nodes", action="store_true")
    parser.add_argument("--clash-guidance", action="store_true")
    parser.add_argument("--clash-guidance-scale", default=0.1, type=float)
    parser.add_argument("--num-integration-steps", default=100, type=int)
    parser.add_argument("--n-nodes-bias", default=5, type=int)
    parser.add_argument("--prior-n-atoms", default="reference",
                        type=str,
                        choices=["conditional", "reference"])
    parser.add_argument("--score-dynamics",
                        default=False, action="store_true")
    parser.add_argument("--score-scale", default=0.5, type=float)
    parser.add_argument("--noise-dynamics",
                        default=False, action="store_true")
    parser.add_argument("--noise-schedule",
                        default="sine", type=str, choices=["sine", "brownian"]
                        )
    parser.add_argument("--noise-scale", default=None)
    parser.add_argument("--dt-pow", default=1.0, type=float)
    parser.add_argument("--qvina2_path", type=str, help="Path to qvina2 executable", required=True)
    parser.add_argument("--discrete-gat",
                        default=False, action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    root_path = Path(args.test_dir)
    sdf_files = sorted(list(root_path.glob("[!.]*.sdf")))

    model_path = args.model_path
    dataset_root = args.dataset_root
    model_type = args.model_type
    mp_idx = args.mp_index
    num_gpus = args.num_gpus
    num_integration_steps = args.num_integration_steps

    model = load_model_dataset_info(model_path, dataset_root, model_type=model_type)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    gen_path = str(save_dir) + f"/chunk_{mp_idx}/"
    os.makedirs(gen_path, exist_ok=True)
    sdf_files_chunks = split_list(sdf_files, num_gpus)

    sdf_chunk = sdf_files_chunks[mp_idx]
    print(f"Processing chunk {mp_idx} with {len(sdf_chunk)} files")
    all_saves = []
    start_time = datetime.now()
    for sdf_file in sdf_chunk:
        print(f"\nProcessing sdf file {sdf_file.stem}")
        save_dicts = wrap_one_sdf_file(
            sdf_file=sdf_file,
            model=model,
            num_integration_steps=num_integration_steps,
            gen_path=gen_path,
            batch_size=args.batch_size,
            pocket_cutoff=args.dist_cutoff,
            nodes_bias=args.n_nodes_bias,
            max_sample_iter=args.max_sample_iter,
            num_ligands_per_pocket_to_sample=args.num_ligands_per_pocket_to_sample,
            prior_n_atoms=args.prior_n_atoms,
            num_processes=args.num_processes,
            score_only=True,
            clash_tolerance=0.5,
            clash_guidance=args.clash_guidance,
            clash_guidance_scale=args.clash_guidance_scale,
            score_dynamics=args.score_dynamics,
            score_scale=args.score_scale,
            noise_dynamics=args.noise_dynamics,
            noise_scale=args.noise_scale,
            dt_pow=args.dt_pow,
            qvina2_path=args.qvina2_path,
            noise_schedule=args.noise_schedule,
            discrete_gat=args.discrete_gat,
        )
        print(save_dicts)
        all_saves.append(save_dicts)

    merged_dict = defaultdict(lambda: defaultdict(list))
    # Iterate through the list and append the values
    for entry in all_saves:
        for key, sub_dict in entry.items():
            for sub_key, value in sub_dict.items():
                merged_dict[key][sub_key].append(value)
    merged_dict = {k: dict(v) for k, v in merged_dict.items()}
    # Save the merged_dict as a JSON file
    save_path = gen_path + "/result_dicts.json"
    with open(save_path, "w") as json_file:
        json.dump(merged_dict, json_file, indent=4)

    print(f"Results saved to {save_path}")
    end = datetime.now() - start_time
    print("Execution time:", end)
    # save arguments
    argsdicts = vars(args)
    argsdicts = {str(k): str(v) for k, v in argsdicts.items()}
    savedirjson = os.path.join(str(args.save_dir), "args.json")
    with open(savedirjson, "w") as f:
        json.dump(argsdicts, f)

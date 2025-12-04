import argparse
import itertools as it
import multiprocessing as mp
import os.path as osp
from pathlib import Path
from typing import List

import MDAnalysis as mda
import numpy as np
import pandas as pd
import prolif as plf
from rdkit import Chem

from e3mol.experiments.sampling.plif_utils.settings import settings


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Prolif Interaction fingerprint computation')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--testset-path', type=str, required=True)
    parser.add_argument('--num-processes', default=1, type=int)
    args = parser.parse_args()
    return args


def wrap_load_sdf_compute_interactions(
    sdf_file: Path,
    test_path: str,
    force_compute: bool = False,
):
    csv_path = str(sdf_file).replace(".sdf", ".csv")
    if not osp.exists(csv_path):
        print("Could not find csv file")
        return None
    else:
        df = pd.read_csv(csv_path)
        if "prolif_interactions" in df.columns and not force_compute:
            print("Prolif interactions have already been computed")
            return None
        else:
            mols = [
                Chem.AddHs(m, addCoords=True)
                for m in Chem.SDMolSupplier(
                    str(sdf_file), removeHs=False, sanitize=True
                )
            ]
            print("Found", len(mols), "molecules")
            assert len(mols) == len(df)
            ref_protein_f = df.pdb_file.tolist()
            assert len(np.unique(ref_protein_f)) == 1
            protein_path = test_path + "/" + ref_protein_f[0] + ".pdb"
            interactions = compute_prolif_interaction(
                ligands=mols,
                protein=protein_path,
                addHs=False,
                return_fp=False,
                settings=settings,
            )
            df["prolif_interactions"] = interactions
            df.to_csv(csv_path, index=False)
            return None


def compute_prolif_interaction(
    ligands: List[Chem.Mol] | str | Path,
    protein: plf.molecule.Molecule | str | Path,
    addHs: bool = True,
    return_fp: bool = False,
    settings=settings,
) -> List[List[str]]:

    if isinstance(protein, str) or isinstance(protein, Path):
        u = mda.Universe(str(protein))
        protein = plf.Molecule.from_mda(u)

    if isinstance(ligands, str) or isinstance(ligands, Path):
        ligands = [
            mol
            for mol in Chem.SDMolSupplier(str(ligands), removeHs=False, sanitize=False)
        ]

    if addHs:
        ligands = [Chem.AddHs(m, addCoords=True) for m in ligands]

    ligands = [plf.molecule.Molecule.from_rdkit(m) for m in ligands]

    # compute ifp
    if settings:
        fp = plf.Fingerprint(
            interactions=settings.interactions,
            parameters=settings.interaction_parameters,
            count=True,
        )
    else:
        fp = plf.Fingerprint(
            count=True,
        )

    fp.run_from_iterable(ligands, protein, progress=False)
    res = fp.to_dataframe()
    interactions = retrieve_interaction_from_prolif_df(res)
    if return_fp:
        interactions = interactions, fp
    return interactions


def retrieve_interaction_from_prolif_df(res: pd.DataFrame):

    if res.shape[0] == 1:
        results = []
        for _, row in res.iterrows():
            for itype in row.index.to_numpy():
                r = "-".join(itype[1:])
                results.append(r)
        results = sorted(results)
    else:
        results = []
        for _, row in res.iterrows():
            interactions = row[row == True]  # type: ignore  # noqa: E712
            subres = []
            for itype in interactions.index.to_numpy():
                r = "-".join(itype[1:])
                subres.append(r)
            subres = sorted(subres)
            results.append(subres)

    return results


def has_exact_interactions(gen_res_its, ref_its, reduce="all") -> float:
    if len(ref_its) == 0:
        return np.nan
    arr = [a in gen_res_its for a in ref_its]
    if reduce == "all":
        out = all(arr)
    elif reduce == "mean":
        out = np.mean(arr)
    return float(out)


def main():
    args = get_args()
    data_path = args.data_path
    num_processes = args.num_processes
    p = Path(data_path)
    all_sdfs = sorted(list(p.glob("**/*.sdf")))
    print(f"Found {len(all_sdfs)} sdf files")
    iterable = zip(
        all_sdfs,
        it.repeat(args.testset_path, len(all_sdfs)),
        it.repeat(False, len(all_sdfs)),
    )
    with mp.Pool(num_processes) as pool:
        _ = pool.starmap(func=wrap_load_sdf_compute_interactions, iterable=iterable)
    print("Done")


if __name__ == "__main__":
    main()

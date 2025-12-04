import os
import os.path
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# from posecheck.utils.strain import calculate_strain_energy
# takes too long, move into another script
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity

from e3mol.experiments.sampling.shape import ShapeTanimotoDistance
from e3mol.experiments.sampling.utils import get_fingerprints_from_smiles_list
from posebusters import PoseBusters


def sdf_to_pdbqt(sdf_file: str, pdbqt_outfile: str, mol_id: int = 0):
    """Converts a sdf file into pdbqt format"""
    cmd = (
        f"obabel {str(sdf_file)} -O {str(pdbqt_outfile)} "
        + f"-f {mol_id + 1} -l {mol_id + 1}"
    )
    _ = subprocess.run(
        [cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    ).stdout
    return pdbqt_outfile


def pdb_to_pdbqt(pdb_file: str | Path, pdbqt_outfile: str | Path, pH: float = 7.4):
    """Converts a pdb file into pdbqt format"""
    cmd = (
        f"obabel {str(pdb_file)} -O {str(pdbqt_outfile)} "
        + f"-xr -p {pH} --partialcharge eem"
    )
    _ = subprocess.run(
        [cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    ).stdout
    return pdbqt_outfile


def extract_score(out, t):
    out_split = out.splitlines()
    if t == "--score_only":
        idx_affinity = [i for i, s in enumerate(out_split) if "Affinity" in s]
        if len(idx_affinity) != 1:
            print("Error in extracting affinity score, return None")
            return np.nan
        affinity_kcal_mol = out_split[idx_affinity[0]]
        affinity_kcal_mol = float(re.findall(r"[-+]?[\d\.\d]+", affinity_kcal_mol)[0])
    else:
        best_idx = out_split.index("-----+------------+----------+----------") + 1
        best_line = out_split[best_idx].split()
        if int(best_line[0]) != 1:
            print("Error in extracting affinity score, return None")
            return np.nan
        affinity_kcal_mol = float(best_line[1])
    return affinity_kcal_mol


def wrap_vina(
    sdf_or_mols: Path | List[Chem.Mol],
    protein_file: Path,
    qvina2_path: str,
    t: str = "--score_only",
    size: float = 30.0,
    exhaustiveness: int = 20,
    return_input_mols: bool = True,
) -> pd.DataFrame:
    affinity_scores = []
    num_heavy_atoms = []

    if protein_file.suffix != ".pdbqt":
        pdbqt_file = protein_file.parent / (protein_file.stem + ".pdbqt")
        if not os.path.exists(str(pdbqt_file)):
            pdb_to_pdbqt(protein_file, pdbqt_file)
    else:
        pdbqt_file = protein_file

    if isinstance(sdf_or_mols, Path) or isinstance(sdf_or_mols, str):
        suppl = Chem.SDMolSupplier(str(sdf_or_mols), sanitize=False, removeHs=False)
        mols = [Chem.AddHs(m, addCoords=True) for m in suppl]
    else:
        mols = [Chem.AddHs(m, addCoords=True) for m in sdf_or_mols]
    td = tempfile.TemporaryDirectory()
    for mol in mols:
        cx, cy, cz = mol.GetConformer().GetPositions().mean(0)
        ligand_pdbqt_file = td.name + "/ligand.pdbqt"
        ligand_tmp_sdf = td.name + "/ligand.sdf"
        writer = Chem.SDWriter(ligand_tmp_sdf)
        writer.write(mol)
        writer.close()
        _ = sdf_to_pdbqt(ligand_tmp_sdf, ligand_pdbqt_file, 0)
        num_heavy_atoms.append(mol.GetNumHeavyAtoms())
        out = os.popen(
            f"/{qvina2_path} --receptor {str(pdbqt_file)} "
            f"--ligand {ligand_pdbqt_file} "
            f"--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} "
            f"--size_x {size} --size_y {size} --size_z {size} "
            f"--exhaustiveness {exhaustiveness} {t}",
        ).read()
        affinity_scores.append(extract_score(out, t))
    td.cleanup()
    outs = np.array(affinity_scores).squeeze(), np.array(num_heavy_atoms).squeeze()
    # shape into length of mols
    if len(outs[0].shape) == 0:
        outs = np.array([outs[0]]), np.array([outs[1]])
    for score, nAtoms, mol in zip(outs[0], outs[1], mols):
        mol.SetProp("score", str(score))
        mol.SetProp("nHeavyAtoms", str(nAtoms))

    out_df = pd.DataFrame()
    if return_input_mols:
        out_df["mols"] = mols
    out_df["score"] = outs[0]
    out_df["nHeavyAtoms"] = outs[1]
    return out_df


def wrap_posebusters_sdf(sdf_file, ref_file, pdb_file) -> pd.DataFrame:
    buster = PoseBusters(config="dock")
    df = buster.bust([sdf_file], ref_file, pdb_file)
    return df


def wrap_posebusters_mols(
    mols, ref_file, pdb_file, return_input_mols: bool = True
) -> pd.DataFrame:
    buster = PoseBusters(config="dock")
    df = buster.bust(mols, ref_file, pdb_file)
    if return_input_mols:
        df["mols"] = mols
    return df


def count_clashes(prot: Chem.Mol, lig: Chem.Mol, tolerance: float = 0.9) -> int:
    """
    Counts the number of clashes between atoms in a protein and a ligand.

    Args:
        prot: RDKit Mol object representing the protein.
        lig: RDKit Mol object representing the ligand.
        tolerance: Distance tolerance for clash detection (default: 0.9).

    Returns:
        clashes: Number of clashes between the protein and the ligand.
    """
    clashes = 0

    try:
        # Get the positions of atoms in the protein and ligand
        prot_pos = prot.GetConformer().GetPositions()
        lig_pos = lig.GetConformer().GetPositions()

        pt = Chem.GetPeriodicTable()

        # Get the number of atoms in the protein and ligand
        num_prot_atoms = prot.GetNumAtoms()
        num_lig_atoms = lig.GetNumAtoms()

        # Calculate the Euclidean distances between all atom pairs in the protein and ligand
        dists = np.linalg.norm(
            prot_pos[:, np.newaxis, :] - lig_pos[np.newaxis, :, :], axis=-1
        )

        # Iterate over the ligand atoms
        for i in range(num_lig_atoms):
            lig_vdw = pt.GetRvdw(lig.GetAtomWithIdx(i).GetAtomicNum())

            # Iterate over the protein atoms
            for j in range(num_prot_atoms):
                prot_vdw = pt.GetRvdw(prot.GetAtomWithIdx(j).GetAtomicNum())

                # Check for clash by comparing the distances with tolerance
                if dists[j, i] + tolerance < lig_vdw + prot_vdw:
                    clashes += 1

    except AttributeError:
        raise ValueError(
            "Invalid input molecules. Please provide valid RDKit Mol objects."
        )

    return clashes


def wrap_compute_clashes(
    sdf_or_mols: Path | List[Chem.Mol],
    pdb_file,
    clash_tolerance: float = 0.5,
) -> np.ndarray:
    if isinstance(sdf_or_mols, Path):
        suppl = Chem.SDMolSupplier(str(sdf_or_mols), sanitize=False, removeHs=False)
        mols = [Chem.AddHs(m, addCoords=True) for m in suppl]
    else:
        mols = [Chem.AddHs(m, addCoords=True) for m in sdf_or_mols]
    prot = Chem.MolFromPDBFile(str(pdb_file), removeHs=False, sanitize=False)
    clashes = []
    for mol in mols:
        cl = count_clashes(prot, mol, tolerance=clash_tolerance)
        clashes.append(cl)
    return np.array(clashes)


def wrap_pose_evaluation_mp(
    mols,
    ref_file: Path,
    pdb_file: Path,
    pdbqt_file: Path,
    qvina2_path: str,
    score_only: bool = True,
    clash_tolerance: float = 0.5,
):
    if score_only:
        t = "--score_only"
    else:
        t = ""
    mols = [Chem.AddHs(m, addCoords=True) for m in mols]

    # PoseBusters
    outs = wrap_posebusters_mols(mols, ref_file, pdb_file)
    outs_pb, mols = outs.drop(["mols"], axis=1), outs["mols"].values
    pb_valids = outs_pb.values.mean(1) == 1.0
    new_mols = []
    if not os.path.exists(str(pdbqt_file)):
        pdbqt_dir = str(pdbqt_file.parent)
        if not os.path.exists(pdbqt_dir):
            os.makedirs(pdbqt_dir)
        pdb_to_pdbqt(pdb_file, pdbqt_file)

    for pb_valid, mol in zip(pb_valids, mols):
        mol.SetProp("pb_valid", str(pb_valid))
        new_mols.append(mol)

    # Vina
    out_df = wrap_vina(
        sdf_or_mols=new_mols, protein_file=pdbqt_file, qvina2_path=qvina2_path, t=t
    )
    out_df["pb_valid"] = pb_valids

    # Steric Clashes
    clashes = wrap_compute_clashes(
        out_df["mols"], pdb_file, clash_tolerance=clash_tolerance
    )
    out_df["clashes"] = clashes
    mols = out_df["mols"]
    new_mols = []

    # Shape Tanimoto Distance
    shape_tanimoto = ShapeTanimotoDistance()
    ref_mol = Chem.SDMolSupplier(str(ref_file), sanitize=False, removeHs=False)[0]
    ref_mol = Chem.AddHs(ref_mol, addCoords=True)
    shape_distances = shape_tanimoto(mols=mols, ref_mols=[ref_mol])
    out_df["shape_tanimoto_distance"] = shape_distances

    # 2d tanimoto similarity to reference
    mols_2d = [Chem.RemoveHs(m) for m in mols]
    for m in mols_2d:
        m.RemoveAllConformers()
    mol2d_smi = [Chem.MolToSmiles(m) for m in mols_2d]
    ref_smi = Chem.MolToSmiles(
        Chem.SDMolSupplier(str(ref_file), sanitize=True, removeHs=True)[0]
    )
    gen_fps = get_fingerprints_from_smiles_list(mol2d_smi)
    ref_fp = get_fingerprints_from_smiles_list([ref_smi])[0]
    sims = np.array(BulkTanimotoSimilarity(ref_fp, gen_fps))
    out_df["sim_to_ref"] = sims

    for mol, pb_valid, cl, sim, shape_dist in zip(
        mols, pb_valids, clashes, sims, shape_distances
    ):
        mol.SetProp("pb_valid", str(pb_valid))
        mol.SetProp("clashes", str(cl))
        mol.SetProp("sim_to_ref", str(sim))
        mol.SetProp("shape_dist_to_ref", str(shape_dist))
        new_mols.append(mol)
    out_df["mols"] = new_mols
    out_df["ref_file"] = ref_file.stem
    out_df["pdb_file"] = pdb_file.stem
    return out_df

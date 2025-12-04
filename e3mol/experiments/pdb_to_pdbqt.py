import glob
import os
import sys


def pdbs_to_pdbqts(pdb_dir, pdbqt_dir):
    if not os.path.exists(pdbqt_dir):
        os.makedirs(pdbqt_dir)
    for file in glob.glob(os.path.join(pdb_dir, "*.pdb")):
        name = os.path.splitext(os.path.basename(file))[0]
        outfile = os.path.join(pdbqt_dir, name + ".pdbqt")
        pdb_to_pdbqt(file, outfile)
        print(f"Wrote converted file to {outfile}")


def pdb_to_pdbqt(pdb_file, pdbqt_file):
    if os.path.exists(pdbqt_file):
        return pdbqt_file
    os.system(f"prepare_receptor4.py -r {pdb_file} -o {pdbqt_file}")
    return pdbqt_file


if __name__ == "__main__":
    pdbs_to_pdbqts(sys.argv[1], sys.argv[2])

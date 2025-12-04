from pathlib import Path
from typing import List, Union

from rdkit import Chem

PathLike = Union[Path, str]


def convert_atomic_element_to_number(element: str):
    pse = Chem.GetPeriodicTable()
    return pse.GetAtomicNumber(element)


def write_sdf_file(sdf_path: PathLike, molecules: List[Chem.Mol]) -> Path:
    sdf_path = Path(sdf_path)
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)
    w.close()
    return sdf_path

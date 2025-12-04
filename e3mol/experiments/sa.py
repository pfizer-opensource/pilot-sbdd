import os
import sys

from rdkit import Chem
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import numpy as np  # noqa: E402
import sascorer  # noqa: E402
import torch  # noqa: E402


def calculate_sa(rdmol: Chem.Mol) -> torch.Tensor:
    sa = np.array([sascorer.calculateScore(Chem.RemoveHs(rdmol))])
    sa = (sa - 1.0) / (10.0 - 1.0)
    sa = 1.0 - sa
    sa = torch.from_numpy(sa).float()
    return sa

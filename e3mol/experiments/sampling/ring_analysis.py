from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def get_ring_info(mol: Chem.Mol) -> Dict[str, int | np.ndarray]:
    ringInfo = mol.GetRingInfo()
    num_rings = ringInfo.NumRings()
    num_fused_rings = sum([ringInfo.IsRingFused(i) for i in range(num_rings)])
    # get fused rings
    num_fused_bonds = [ringInfo.NumFusedBonds(i) for i in range(num_rings)]
    # the integer means how many fused bonds are shared with other rings.
    # 0 means a ring is not used,
    # 1 means 1 bond is shared with another ring, i.e. 2-fused-ring-syste
    # 2 means 2 bonds are shared with 2 other rings, i.e. 3-fused-ring-system
    # 3 means 3 bonds are shared with 3 other rings, i.e. 4-fused-ring-system
    # etc.
    # we offset by 1 to make it clear
    fused_ring_systems = 1 + np.array(num_fused_bonds)

    out = {
        "num_rings": num_rings,
        "num_fused_rings": num_fused_rings,
        "k_ring_systems": fused_ring_systems,  # includes to query ring itself
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "num_aromatic_heterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        "num_aliphatic_heterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles(
            mol
        ),
    }

    return out

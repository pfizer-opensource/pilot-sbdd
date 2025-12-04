import logging
import random
from copy import deepcopy
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, Recap
from torch_geometric.data import Data

from e3mol.experiments.data.smarts_aliases import cut_smarts_aliases_by_name

logger = logging.getLogger(__name__)


def get_random_fragment_anchor_mask(
    fragments: List[List[int]], anchors: List[List[int]], natoms: int
) -> np.ndarray:

    # USED in training

    fixed_mask = np.zeros(natoms)
    select = random.randint(0, len(fragments) - 1)
    # note that if the molecule consist only of 1 fragment, i.e. len(fragments) - 1 == 0,
    # we keep the entire molecule as variable
    selected_frag = fragments[select]
    fixed_mask[selected_frag] = 1
    anchor_mask = np.zeros(natoms)
    selected_anchor = anchors[select]
    if len(selected_anchor) > 0:
        anchor_mask[selected_anchor] = 1

    mask = np.stack([fixed_mask, anchor_mask], axis=1)

    return mask


# Preprocessing before training


def get_unique_fragments(frags=List[Chem.Mol], removeHs: bool = True) -> List[Chem.Mol]:
    unique_frags = []
    for f in frags:
        unique_frags.append(Chem.MolToSmiles(f))
    unique_frags = list(set(unique_frags))
    unique_frags = [Chem.MolFromSmiles(s) for s in unique_frags]
    if not removeHs:
        unique_frags = [Chem.AddHs(m) for m in unique_frags]
    return unique_frags


def get_briccs_fragmentation(
    mol: Chem.Mol,
    minFragmentSize: int = 4,
    remove_duplicates: bool = True,
    removeHs: bool = True,
) -> List[Chem.Mol]:
    fragmented = BRICS.BreakBRICSBonds(mol)
    pieces = Chem.GetMolFrags(fragmented, asMols=True)
    frags = [p for p in pieces if p.GetNumAtoms() >= minFragmentSize]
    if len(frags) == 0:
        frags = [mol]
    else:
        if remove_duplicates:
            frags = get_unique_fragments(frags, removeHs=removeHs)
    return frags


def get_recap_fragmentation(
    mol: Chem.Mol,
    minFragmentSize: int = 4,
    remove_duplicates: bool = True,
    removeHs: bool = True,
) -> List[Chem.Mol]:
    pieces = Recap.RecapDecompose(mol, minFragmentSize=minFragmentSize)
    frags = [c.mol for c in pieces.children.values()]
    if len(frags) == 0:
        frags = [mol]
    else:
        if remove_duplicates:
            frags = get_unique_fragments(frags, removeHs=removeHs)
    return frags


def get_cuttable_fragmentation(
    mol: Chem.Mol,
    num_cuts: int = 1,
    cut_type: str = "cut_Amides",
    small_only: bool = True,
    minFragmentSize: int = 3,
    remove_duplicates: bool = True,
    removeHs: bool = True,
) -> List[Chem.Mol]:

    if not removeHs:
        logger.log("Warning: removeHs is not supported for cuttable fragmentation")
        raise ValueError

    cut_pattern = Chem.MolFromSmarts(cut_smarts_aliases_by_name[cut_type].smarts)
    atom_pairs = mol.GetSubstructMatches(cut_pattern, uniquify=True)

    pieces_all = []
    for i, first_pair in enumerate(atom_pairs):
        a1, a2 = first_pair
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond.IsInRing():
            continue
        fragmented = Chem.FragmentOnBonds(mol, [bond.GetIdx()])
        pieces = Chem.GetMolFrags(fragmented, asMols=True)

        pieces = sorted(pieces, key=lambda x: len(x.GetAtoms()))
        size_min = min([len(p.GetAtoms()) for p in pieces]) - 1
        if small_only:
            pieces = pieces[0:]

        for p in pieces:
            if len(p.GetAtoms()) - 1 >= minFragmentSize:
                pieces_all.append(p)

        if num_cuts >= 2 and size_min > minFragmentSize:
            for _, second_pair in enumerate(atom_pairs[i + 1 :], i + 1):
                a3, a4 = second_pair
                bond2 = mol.GetBondBetweenAtoms(a3, a4)
                if bond2.IsInRing():
                    continue

                fragmented = Chem.FragmentOnBonds(mol, [bond.GetIdx(), bond2.GetIdx()])
                pieces = Chem.GetMolFrags(fragmented, asMols=True)

                for p in pieces:
                    if len(p.GetAtoms()) - 2 >= minFragmentSize:
                        num_anchors = 0
                        for a in p.GetAtoms():
                            if a.GetAtomicNum() == 0:
                                num_anchors += 1

                        if num_anchors == 2:
                            pieces_all.append(p)
                            break

    if len(pieces_all) == 0:
        pieces_all = [mol]

    pieces_smiles = [Chem.MolToSmiles(p, isomericSmiles=False) for p in pieces_all]
    if remove_duplicates:
        pieces_smiles = set(pieces_smiles)

    pieces_all = [Chem.MolFromSmiles(m) for m in pieces_smiles]

    if not removeHs:
        pieces_all = [Chem.AddHs(m) for m in pieces_all]

    return pieces_all


def get_subtructure_matches(mol, frag):

    for atom in frag.GetAtoms():
        atom.SetIsotope(0)
    params = Chem.AdjustQueryParameters()
    params.makeBondsGeneric = False
    params.makeDummiesQueries = True
    frag_copy = Chem.AdjustQueryProperties(deepcopy(frag), params)
    check = mol.HasSubstructMatch(frag_copy)
    if check:
        matches = mol.GetSubstructMatches(frag_copy)
    else:
        params = Chem.AdjustQueryParameters()
        params.makeBondsGeneric = True
        params.makeDummiesQueries = True
        frag_copy = Chem.AdjustQueryProperties(deepcopy(frag), params)
        check = mol.HasSubstructMatch(frag_copy)
        assert check
        matches = mol.GetSubstructMatches(frag_copy)
    return matches


def get_atom_ids_and_anchors(mol: Chem.Mol, frag: Chem.Mol) -> Dict[str, List[int]]:
    # mol: Full RDKIT Molecule
    # frag: Fragment RDKIT Molecule
    matches = get_subtructure_matches(mol=mol, frag=frag)
    anchor_ids = []
    sub_ids = []
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            # dummy atom in the fragment determines the anchor
            anchor_ids.append(atom.GetIdx())
        else:
            sub_ids.append(atom.GetIdx())
    if len(matches) > 1:
        # one fragment appears more often, e.g. a phenyl
        # or highly symmetric molecules
        # make sure the ids
        matches = random.choice(matches)
    else:
        matches = matches[0]
    # select the sub_ids and anchor_ids
    sub_ids = [matches[i] for i in sub_ids]
    anchor_ids = [matches[i] for i in anchor_ids]
    out = {"ids": sub_ids, "anchors": anchor_ids}
    return out


class FragmentTransform:
    def __init__(
        self,
        minFragmentSize: int = 4,
        method: str = "recap",
        removeHs=True,
        include_middle_fragment: bool = True,
        cuttable_pattern: str = "cut_Amides",
        small_only: bool = True,  # used in cuttable fragmentation
    ):
        super().__init__()
        self.minFragmentSize = minFragmentSize
        self.method = method
        self.removeHs = removeHs
        self.include_middle_fragment = include_middle_fragment
        self.cuttable_pattern = cuttable_pattern
        self.small_only = small_only
        assert method in ["recap", "briccs", "recap-briccs", "briccs-recap", "cuttable"]

    def __call__(self, data: Data) -> Data:
        mol = data.mol
        method_ = None
        if self.method == "recap":
            pieces = get_recap_fragmentation(
                mol,
                self.minFragmentSize,
                remove_duplicates=True,
                removeHs=self.removeHs,
            )
        elif self.method == "briccs":
            pieces = get_briccs_fragmentation(
                mol,
                self.minFragmentSize,
                remove_duplicates=True,
                removeHs=self.removeHs,
            )
        elif self.method in ["recap-briccs", "briccs-recap"]:
            if random.random() > 0.5:
                pieces = get_recap_fragmentation(
                    mol,
                    self.minFragmentSize,
                    remove_duplicates=True,
                    removeHs=self.removeHs,
                )
                method_ = "recap"
                if (
                    len(pieces) == 1
                ):  # if recap decomposition returns the entire molecule
                    pieces = get_briccs_fragmentation(
                        mol,
                        self.minFragmentSize,
                        remove_duplicates=True,
                        removeHs=self.removeHs,
                    )
                    method_ = "briccs"
            else:
                pieces = get_briccs_fragmentation(
                    mol,
                    self.minFragmentSize,
                    remove_duplicates=True,
                    removeHs=self.removeHs,
                )
                method_ = "briccs"
        elif self.method == "cuttable":
            if self.include_middle_fragment:
                num_cuts = 2
            else:
                num_cuts = 1

            pieces = get_cuttable_fragmentation(
                mol,
                minFragmentSize=self.minFragmentSize,
                num_cuts=num_cuts,
                cut_type=self.cuttable_pattern,
                small_only=self.small_only,
            )
            method_ = "cuttable"

        sub_ids = []
        anchors = []
        for m in pieces:
            r = get_atom_ids_and_anchors(frag=m, mol=mol)
            sub_ids.append(r["ids"])
            anchors.append(r["anchors"])
        data.sub_ids = sub_ids
        data.anchor_ids = anchors
        data.method = method_ if method_ is not None else self.method
        return data

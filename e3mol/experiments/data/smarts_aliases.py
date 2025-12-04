class CutSmarts:
    def __init__(self, name, smarts, description):
        self.name = name
        self.smarts = smarts
        self.description = description


cut_smarts_aliases_by_name = {}

cut_smarts_aliases = [
    CutSmarts(
        "default",
        "[#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#1;!$([CH2]);!$([CH3][CH2])]",
        "Cut all C-[!H] non-ring single bonds except for Amides/Esters/Amidines/Sulfonamides "
        "and CH2-CH2 and CH2-CH3 bonds",
    ),
    CutSmarts(
        "cut_AlkylChains",
        "[#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#1]",
        "As default, but also cuts CH2-CH2 and CH2-CH3 bonds",
    ),
    CutSmarts(
        "cut_Amides",
        "[#6+0]!@!=!#[!#0;!#1;!$([CH2]);!$([CH3][CH2])]",
        "As default, but also cuts [O,N]=C-[O,N] single bonds",
    ),
    CutSmarts(
        "cut_all",
        "[#6+0]!@!=!#[!#0;!#1]",
        "Cuts all Carbon-[!H] single non-ring bonds.\
         Use carefully, this will create a lot of cuts",
    ),
    CutSmarts("exocyclic", "[R]!@!=!#[!#0;!#1]", "Cuts all exocyclic single bonds"),
    CutSmarts(
        "exocyclic_NoMethyl",
        "[R]!@!=!#[!#0;!#1;!$([CH3])]",
        "Cuts all exocyclic single bonds apart from those connecting to CH3 groups",
    ),
]


for alias in cut_smarts_aliases:
    cut_smarts_aliases_by_name[alias.name] = alias

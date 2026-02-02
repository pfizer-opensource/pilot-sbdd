# Coupled Fragment-Based Generative Modeling with Stochastic Interpolants
Official implementation of the pre-print "Coupled Fragment-Based Generative Modeling with Stochastic Interpolants" by Tuan Le, Yanfei Guan, Djork-Arné Clevert and Kristof T. Schütt.


## Installation

```
# clone the repository and cd directory to install
git clone git@github.com:pfizer-opensource/pilot-sbdd.git && cd pilot-sbdd
mamba env create -f environment.yaml
mamba activate pilot-sbdd
pip install torch_geometric==2.4.0
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install -e .
pip install useful_rdkit_utils
```

Please check if the CUDA versions of pytorch and torch_geometric are correctly installed. On a compute node with GPU, test the following:

```bash
mamba activate pilot-sbdd
python -c "import torch; import torch_geometric; print(torch.cuda.is_available())"
````

## Implemented models
- Diffusion Models (unconditional, conditional, mixed)
- Flow Models  (unconditional, conditional, mixed)

## Publicly available datasets used in this study
- CrossDocked2020
- KinoData3d  

The paths will decompress into 
```
datasets/
└── crossdocked2020/
    ├── raw/
    ├── processed/
    ├── test/
└── kinodata3d/
    ├── raw/
    ├── processed/
    ├── test/
```

Datasets can be downloaded here: https://figshare.com/articles/dataset/Datasets_including_CrossDocked2020_and_KinoData-3D/30739232?file=59947856

## Training
#### Training on Kinodata3d

Set arguments or put into config.

To train the diffusion model (de-novo, conditional fragment only, or de-novo and conditional fragment mixed) run:
```bash
cd e3mol/experiments
save_dir=""
dataset_root=""

# unconditional, learning p(molecule | pocket), diffusion model
python run_train.py --conf configs/diffusion_kinodata.yaml --save-dir $save_dir --dataset-root $dataset_root --batch-size 16 --gpus 1

# conditional, learning p(variable_fragment | fixed_fragment, pocket)
python run_train.py --conf configs/diffusion_kinodata.yaml --save-dir $save_dir --dataset-root $dataset_root --batch-size 16 --gpus 1 --node-level-t --fragmentation

# unconditional, learning p(molecule | pocket) and  conditional, learning p(variable_fragment | fixed_fragment, pocket) 

python run_train.py --conf configs/diffusion_kinodata.yaml --save-dir $save_dir --dataset-root $dataset_root --batch-size 16 --gpus 1 --node-level-t --fragmentation --fragmentation-mix
```


To train the flow model (de-novo, conditional fragment only, or de-novo and conditional fragment mixed) run:
```bash
cd e3mol/experiments
save_dir=""
dataset_root=""

# unconditional, learning p(molecule | pocket), diffusion model
python run_train.py --conf configs/flow_kinodata.yaml --save-dir $save_dir --dataset-root $dataset_root --batch-size 16 --gpus 1

# conditional, learning p(variable_fragment | fixed_fragment, pocket)
python run_train.py --conf configs/flow_kinodata.yaml --save-dir $save_dir --dataset-root $dataset_root --batch-size 16 --gpus 1 --node-level-t --fragmentation

# unconditional, learning p(molecule | pocket) and  conditional, learning p(variable_fragment | fixed_fragment, pocket) 

python run_train.py --conf configs/flow_kinodata.yaml --save-dir $save_dir --dataset-root $dataset_root --batch-size 16 --gpus 1 --node-level-t --fragmentation --fragmentation-mix
```

## Model weights

We provide model weights for the diffusion and flow models trained on Kinodata3d here

-----

Please download model weights from the kinodata3d model here: https://figshare.com/articles/dataset/Checkpoints_to_lightning_model/30739268?file=59948612

The paths will decompress into 
```
ckpts/
└── kinodata3d/
    ├── diffusion/
    ├── flow/
    │   ├── recap/
    │   ├── brics/
    │   └── cutable/
    │       └── best_valid.ckpt
```


Both checkpoints can be used for de-novo generation of fragment/core replacement as illustrated in the accompanying notebook below.
## Inference
- See example notebook `inference_notebooks/generate_molecules_pl.ipynb`

## Reference
If you make use of this repository, please consider citing the following works

```
@UNPUBLISHED{Le2025-re,
title    = "Coupled fragment-based generative modeling with stochastic interpolants",
author   = "Le, Tuan and Guan, Yanfei and Clevert, Djork-Arné and Schütt, Kristof T",
journal  = "ChemRxiv",
month    =  oct,
year     =  2025
}
```

and 

```
@Article{cremer2024pilotequivariantdiffusionpocket,
author ="Cremer, Julian and Le, Tuan and Noé, Frank and Clevert, Djork-Arné and Schütt, Kristof T.",
title  ="PILOT: equivariant diffusion for pocket-conditioned de novo ligand generation with multi-objective guidance via importance sampling",
journal  ="Chem. Sci.",
year  ="2024",
volume  ="15",
issue  ="36",
pages  ="14954-14967",
publisher  ="The Royal Society of Chemistry",
doi  ="10.1039/D4SC03523B",
url  ="http://dx.doi.org/10.1039/D4SC03523B"
}
```


## Note
This repository is a refactored and extended version of the following repository:
https://github.com/pfizer-opensource/e3moldiffusion.

The changes mainly include the code implementation of the flow matching and extending the learning to conditional fragment learning by leveraging fragmentation algorithm to obtain fixed/variable masks.

## Contact
Should you face installation issues or have any other questions, feel free to contact via email  tuan.le@pfizer.com
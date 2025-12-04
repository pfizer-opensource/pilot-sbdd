import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter_mean

from e3mol.experiments.data.datainfo import DatasetInfo
from e3mol.experiments.data.fragmentation import get_random_fragment_anchor_mask
from e3mol.experiments.diffmodel import EQGATDiffL, EQGATDiffPL
from e3mol.experiments.flowmodel import EQGATDiffL as EQGATDiffLFlow
from e3mol.experiments.flowmodel import EQGATDiffPL as EQGATDiffPLFlow
from e3mol.experiments.losses import DiffusionLoss, ligand_pocket_clash_energy_loss
from e3mol.experiments.sampling.analyze import analyze_stability_for_molecules


class UniformTimeSampler:
    def sample(self, sample_shape):
        return torch.rand(sample_shape)


def check_validity(rdmol: Chem.Mol) -> Optional[Chem.Mol]:
    """Checks if the generated rdkit mole is valid
    # Currently checks if the molecule is connected and can be sanitized
    Args:
        rdmol (Chem.Mol): generated rdkit molecule
    Returns:
        Optional[Chem.Mol]: the rdkit mol if valid, else None
    """
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
        if len(mol_frags) > 1:
            return None
        else:
            Chem.SanitizeMol(rdmol)
            return rdmol
    except Exception:
        return None


class Trainer(pl.LightningModule):

    def __init__(
        self,
        hparams: Dict[str, Any],
        dataset_info: DatasetInfo,
        ckpt_path: Optional[str] = None,
        pocket_noise_std: float = 0.1,
        smiles_train: Optional[List[str]] = None,
    ):

        super().__init__()

        if "model_type" not in hparams.keys():
            hparams["model_type"] = "protein-ligand"
        if "addNumHs" not in hparams.keys():
            hparams["addNumHs"] = False
        if "model" not in hparams.keys():
            hparams["model"] = "diffusion"
        if "fragmentation" not in hparams.keys():
            hparams["fragmentation"] = False
        if "fragmentation_mix" not in hparams.keys():
            hparams["fragmentation_mix"] = False
        if "fragment_prior" not in hparams.keys():
            hparams["fragment_prior"] = "fragment"  # redundant
        if "addHybridization" not in hparams.keys():
            hparams["addHybridization"] = False
        if "t_cond_frac" not in hparams.keys():
            hparams["t_cond_frac"] = 0.5

        self.save_hyperparameters(hparams)
        self.dataset_info = dataset_info

        if hparams["model_type"] == "ligand":
            print("Using Ligand Diffusion Model")
            self.model = EQGATDiffL(hparams, dataset_info, ckpt_path)
        elif hparams["model_type"] == "protein-ligand":
            print("Using Protein-Ligand Diffusion Model")
            self.model = EQGATDiffPL(
                hparams, dataset_info, ckpt_path, uniform_prior=False
            )
        else:
            raise ValueError(
                "Invalid model type. Only ligand or protein-ligand supported"
            )

        self.latent_net = None  # type: ignore [assignment]
        self.latent_loss = None  # type: ignore [assignment]

        modalities = ["coords", "atoms", "charges", "bonds"]
        if self.hparams.addNumHs:
            modalities.append("numHs")
        if self.hparams.addHybridization:
            modalities.extend(["hybridization"])
        self.diffusion_loss = DiffusionLoss(
            modalities=modalities,
            param=["data"] * len(modalities),
        )
        self.pocket_noise_std = pocket_noise_std
        if "node_level_t" in hparams.keys():
            self.node_level_t = hparams["node_level_t"]
        else:
            self.node_level_t = False

        self.i = 0
        self.validity = 0.0
        self.connected_components = 0.0
        self.angles_w1 = 1000.0
        self.smiles_train = smiles_train

    def forward(
        self,
        batch: Batch,
        t: Tensor,
        latent_gamma: float = 1.0,
        fragment_anchor_mask: Optional[Tensor] = None,
    ):

        out = self.model(
            batch=batch,
            t=t,
            z=None,
            latent_gamma=latent_gamma,
            pocket_noise_std=self.pocket_noise_std,
            fragment_anchor_mask=fragment_anchor_mask,
        )

        return out

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self, log=False):
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:
            if self.local_rank == 0:
                print(f"Running evaluation in epoch {self.current_epoch + 1}")

            if self.hparams.fragmentation:
                if self.local_rank == 0:
                    print("Running conditional evaluation")
                final_res = self.run_evaluation_conditional(
                    step=self.i,
                    verbose=True,
                    device="cuda",
                )
                if log:
                    self.log(
                        name="val/validity_cond",
                        value=final_res.validity.tolist()[0],
                        on_epoch=True,
                        sync_dist=True,
                    )
                    self.log(
                        name="val/uniqueness_cond",
                        value=final_res.uniqueness.tolist()[0],
                        on_epoch=True,
                        sync_dist=True,
                    )
                    self.log(
                        name="val/novelty_cond",
                        value=final_res.novelty.tolist()[0],
                        on_epoch=True,
                        sync_dist=True,
                    )

            if self.local_rank == 0:
                print("Running unconditional evaluation")

            final_res = self.run_evaluation_unconditional(
                step=self.i,
                dataset_info=self.dataset_info,
                verbose=True,
                inner_verbose=False,
                eta_ddim=1.0,
                ddpm=True,
                every_k_step=1,
                device="cuda",
                prior_n_atoms="conditional",
                save_traj=False,
            )
            self.i += 1
            if log:
                self.log(
                    name="val/validity",
                    value=final_res.validity.tolist()[0],
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    name="val/uniqueness",
                    value=final_res.uniqueness.tolist()[0],
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    name="val/novelty",
                    value=final_res.novelty.tolist()[0],
                    on_epoch=True,
                    sync_dist=True,
                )

    @torch.no_grad()
    # TODO: refactor function signature
    def run_evaluation_unconditional(
        self,
        step: int,
        dataset_info,
        save_dir: Optional[str] = None,
        return_molecules: bool = False,
        verbose: bool = False,
        inner_verbose=False,
        save_traj=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        use_ligand_dataset_sizes: bool = False,
        build_obabel_mol: bool = False,
        run_test_eval: bool = False,
        guidance_scale: float = 1.0e-4,
        property_classifier_guidance=None,
        property_classifier_guidance_complex=False,
        property_classifier_self_guidance=False,
        classifier_guidance_scale=None,
        ckpt_property_model: Optional[str] = None,
        n_nodes_bias: int = 0,
        device: str = "cpu",
        encode_ligand: bool = False,
        prior_n_atoms: str = "reference",
        **kwargs,
    ):
        """
        Runs the evaluation on the entire validation dataloader.
        Generates 1 ligand in 1 receptor structure
        """

        dataloader = (
            self.trainer.datamodule.val_dataloader()
            if not run_test_eval
            else self.trainer.datamodule.test_dataloader()
        )
        molecule_list = []
        start = datetime.now()

        for i, data in enumerate(dataloader):

            try:
                num_graphs = len(data.batch.bincount())
                if use_ligand_dataset_sizes or prior_n_atoms == "reference":
                    num_nodes_lig = data.batch.bincount().to(self.device)
                elif prior_n_atoms == "conditional":
                    if self.hparams.model_type == "protein-ligand":
                        num_nodes_lig = self.dataset_info.conditional_size_distribution.sample_conditional(  # noqa: B950
                            n1=None, n2=data.pos_pocket_batch.bincount()
                        ).to(
                            self.device
                        )
                    else:
                        num_nodes_lig = self.dataset_info.nodes_dist.sample_n(
                            n_samples=num_graphs, device=self.device
                        ).to(self.device)
                    num_nodes_lig += n_nodes_bias
                else:
                    raise ValueError("Invalid prior_n_atoms")

                molecules, _, cl_loss = self.model.reverse_sampling(
                    num_graphs=num_graphs,
                    device=self.device,
                    num_nodes_lig=num_nodes_lig,
                    pocket_data=data,
                    verbose=inner_verbose,
                    save_traj=save_traj,
                    ddpm=ddpm,
                    eta_ddim=eta_ddim,
                    every_k_step=every_k_step,
                    ckpt_property_model=ckpt_property_model,
                    property_classifier_guidance=property_classifier_guidance,
                    property_classifier_guidance_complex=property_classifier_guidance_complex,
                    property_classifier_self_guidance=property_classifier_self_guidance,
                    classifier_guidance_scale=classifier_guidance_scale,
                    save_dir=save_dir,
                    build_obabel_mol=build_obabel_mol,
                    iteration=i,
                    encode_ligand=encode_ligand,
                    z=None,
                    # anchor_mask=None,
                )
                molecule_list.extend(molecules)
                del data
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue

        # evaluate all generated samples from the validation set
        try:
            outs = analyze_stability_for_molecules(
                dataset_info=self.dataset_info,
                device=device,
                smiles_train=self.smiles_train,
                molecule_list=molecule_list,
                remove_hs=self.hparams.remove_hs,
                local_rank=self.local_rank,
                test=run_test_eval,
                filter_by_lipinski=False,
                filter_by_posebusters=False,
                pdb_file=None,
            )
        except Exception as e:
            print(f"Error in analyze_stability_for_molecules: {e}")
            outs = {
                "validity_dict": {"validity": 0.0},
                "statistics_dict": {
                    "connected_components": 0.0,
                    "sampling/AnglesW1": 1000.0,
                },
                "stability_dict": {"stability": 0.0},
            }

        if not run_test_eval:
            save_cond = (
                self.validity < outs["validity_dict"]["validity"]
                and self.connected_components
                <= outs["statistics_dict"]["connected_components"]
            )
            save_cond = (
                save_cond
                or outs["statistics_dict"]["sampling/AnglesW1"] < self.angles_w1
            )
        else:
            save_cond = False

        if save_cond:
            self.validity = outs["validity_dict"]["validity"]
            self.connected_components = outs["statistics_dict"]["connected_components"]
            self.angles_w1 = outs["statistics_dict"]["sampling/AnglesW1"]
            save_path = os.path.join(self.hparams.save_dir, "best_valid.ckpt")
            self.trainer.save_checkpoint(save_path)
        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")
        total_res_dict = dict(outs["stability_dict"])
        total_res_dict.update(outs["validity_dict"])
        total_res_dict.update(outs["statistics_dict"])
        total_res_dict.update({"clash_loss": cl_loss})

        total_res: pd.DataFrame = pd.DataFrame.from_dict([total_res_dict])
        if self.local_rank == 0:
            print(total_res)
        total_res["step"] = str(step)
        total_res["epoch"] = str(self.current_epoch)
        total_res["run_time"] = str(run_time)
        if save_dir is None:
            save_dir = os.path.join(
                self.hparams.save_dir,
                "run" + str(self.hparams.id),
                "evaluation.csv",
            )
        else:
            save_dir = os.path.join(save_dir, "evaluation.csv")
        if self.local_rank == 0:
            with open(save_dir, "a") as f:
                total_res.to_csv(f, header=True)
            print(f"Saving evaluation csv file to {save_dir}")

        return total_res

    @torch.no_grad()
    # TODO: refactor function signature
    def run_evaluation_conditional(
        self,
        step: int = 0,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        inner_verbose=False,
        save_traj=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        build_obabel_mol: bool = False,
        run_test_eval: bool = False,
        device: str = "cpu",
        encode_ligand: bool = False,
        **kwargs,
    ):
        """
        Runs the evaluation on the entire validation dataloader.
        Generates 1 ligand in 1 receptor structure
        """

        dataloader = (
            self.trainer.datamodule.val_dataloader()
            if not run_test_eval
            else self.trainer.datamodule.test_dataloader()
        )
        molecule_list = []
        start = datetime.now()

        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            batch_num_nodes = data.batch.bincount()
            fragment_anchor_mask = (
                torch.from_numpy(
                    np.concatenate(
                        [
                            get_random_fragment_anchor_mask(
                                frag_ids, anchor_ids, n.item()
                            )
                            for frag_ids, anchor_ids, n in zip(
                                data.sub_ids, data.anchor_ids, batch_num_nodes
                            )
                        ]
                    )
                )
                .to(data.x.device)
                .float()
            )
            n = data.x.shape[0]
            data.data_mask = torch.ones((n,), device=data.x.device, dtype=torch.bool)
            data.lig_inpaint_mask = ~(fragment_anchor_mask[:, 0].bool())  # kept fixed
            data.anchor_mask = fragment_anchor_mask[:, 1]
            data.n_variable = scatter_add(
                fragment_anchor_mask[:, 0],
                data.batch,
                dim=0,
                dim_size=len(batch_num_nodes),
            )

            molecules, _, cl_loss = self.model.reverse_sampling_node_level_t(
                pocket_data=data,
                device=self.device,
                verbose=inner_verbose,
                save_traj=save_traj,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                save_dir=save_dir,
                build_obabel_mol=build_obabel_mol,
                iteration=i,
                encode_ligand=encode_ligand,
                z=None,
            )
            molecule_list.extend(molecules)
            del data
            torch.cuda.empty_cache()

        # evaluate all generated samples from the validation set
        try:
            outs = analyze_stability_for_molecules(
                dataset_info=self.dataset_info,
                device=device,
                smiles_train=self.smiles_train,
                molecule_list=molecule_list,
                remove_hs=self.hparams.remove_hs,
                local_rank=self.local_rank,
                test=run_test_eval,
                filter_by_lipinski=False,
                filter_by_posebusters=False,
                pdb_file=None,
            )
        except Exception as e:
            print(f"Error in analyze_stability_for_molecules: {e}")
            outs = {
                "validity_dict": {"validity": 0.0},
                "statistics_dict": {
                    "connected_components": 0.0,
                    "sampling/AnglesW1": 1000.0,
                },
                "stability_dict": {"stability": 0.0},
            }

        if not run_test_eval:
            save_cond = (
                self.validity < outs["validity_dict"]["validity"]
                and self.connected_components
                <= outs["statistics_dict"]["connected_components"]
            )
            save_cond = (
                save_cond
                or outs["statistics_dict"]["sampling/AnglesW1"] < self.angles_w1
            )
        else:
            save_cond = False

        if save_cond:
            self.validity = outs["validity_dict"]["validity"]
            self.connected_components = outs["statistics_dict"]["connected_components"]
            self.angles_w1 = outs["statistics_dict"]["sampling/AnglesW1"]
            save_path = os.path.join(self.hparams.save_dir, "best_valid_cond.ckpt")
            self.trainer.save_checkpoint(save_path)
        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")
        total_res_dict = dict(outs["stability_dict"])
        total_res_dict.update(outs["validity_dict"])
        total_res_dict.update(outs["statistics_dict"])
        total_res_dict.update({"clash_loss": cl_loss})

        total_res: pd.DataFrame = pd.DataFrame.from_dict([total_res_dict])
        if self.local_rank == 0:
            print(total_res)
        total_res["step"] = str(step)
        total_res["epoch"] = str(self.current_epoch)
        total_res["run_time"] = str(run_time)
        if save_dir is None:
            save_dir = os.path.join(
                self.hparams.save_dir,
                "run" + str(self.hparams.id),
                "evaluation_conditional.csv",
            )
        else:
            save_dir = os.path.join(save_dir, "evaluation_conditional.csv")
        if self.local_rank == 0:
            with open(save_dir, "a") as f:
                total_res.to_csv(f, header=True)
            print(f"Saving evaluation csv file to {save_dir}")

        return total_res

    def step_fnc(self, batch, batch_idx, stage: str):

        batch.batch = batch.pos_batch
        batch_size = int(batch.batch.max()) + 1
        batch_num_nodes = batch.batch.bincount()
        if self.node_level_t:
            N = batch.x.size(0)
            t_batch = torch.randint(
                low=1,
                high=self.hparams.timesteps + 1,
                size=(batch_size,),
                device=batch.x.device,
            ).repeat_interleave(batch_num_nodes)
            if not self.hparams.fragmentation:
                # just mix all node levels
                mask_batch = (
                    torch.rand((batch_size, 1), device=batch.x.device)
                    < (1.0 - self.hparams.t_cond_frac)
                ).repeat_interleave(batch_num_nodes)
                t_node = torch.randint(
                    low=1,
                    high=self.hparams.timesteps + 1,
                    size=(N,),
                    device=batch.x.device,
                )
                t = (
                    t_batch * mask_batch.float() + t_node * (1 - mask_batch.float())
                ).long()
                fragment_anchor_mask = None
            else:
                # conditional models
                fragment_anchor_mask = (
                    torch.from_numpy(
                        np.concatenate(
                            [
                                get_random_fragment_anchor_mask(
                                    frag_ids, anchor_ids, n.item()
                                )
                                for frag_ids, anchor_ids, n in zip(
                                    batch.sub_ids, batch.anchor_ids, batch_num_nodes
                                )
                            ]
                        )
                    )
                    .to(batch.x.device)
                    .float()
                )
                if self.hparams.fragmentation_mix:
                    t_variable = (
                        torch.rand((batch_size, 1), device=batch.x.device)
                        < (1.0 - self.hparams.t_cond_frac)
                    ).repeat_interleave(batch_num_nodes)
                    # t_variable means for unconditional learning
                    t_variable = t_variable.float()
                    # t_context means for conditional learning
                    t_context = 1.0 - t_variable
                    ones = torch.ones_like(t_batch)
                    t = (
                        t_batch
                        * (
                            t_variable * ones.float()
                            + t_context * fragment_anchor_mask[:, 0].float()
                        )
                    ).long()
                    t_ = torch.empty_like(fragment_anchor_mask)
                    t_[:, 0] = 1  # fragment 1 means all is variable, 0 means fixed
                    t_[:, 1] = 0  # anchor 0 shows no anchor
                    fragment_anchor_mask = (
                        t_context.unsqueeze(-1) * fragment_anchor_mask.float()
                        + t_variable.unsqueeze(-1) * t_.float()
                    )
                else:
                    t = (t_batch * fragment_anchor_mask[:, 0].float()).long()
                t[t == 0] = (
                    1
                    # timestep 0 is actually the discrete datastate
                    # but due to the way the model is implemented,
                    # we need to start at 1,
                )
                # the prior state is T = 500 by default.
        else:
            t = torch.randint(
                low=1,
                high=self.hparams.timesteps + 1,
                size=(batch_size,),
                dtype=torch.long,
                device=batch.x.device,
            )
            fragment_anchor_mask = None

        if self.hparams.loss_weighting == "snr_s_t":
            weights = self.model.sde_atom_charge.snr_s_t_weighting(
                s=t - 1, t=t, device=self.device, clamp_min=0.05, clamp_max=1.5
            )
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.model.sde_atom_charge.snr_t_weighting(
                t=t,
                device=self.device,
                clamp_min=0.05,
                clamp_max=1.5,
            )
        elif self.hparams.loss_weighting == "uniform":
            weights = None
        else:
            raise NotImplementedError

        out_dict = self(
            batch=batch,
            t=t,
            latent_gamma=1.0,
            fragment_anchor_mask=fragment_anchor_mask,
        )

        true_data = {
            "coords": out_dict["coords_true"],
            "atoms": out_dict["atoms_true"],
            "charges": out_dict["charges_true"],
            "bonds": out_dict["bonds_true"],
            "numHs": out_dict["numHs_true"],
            "hybridization": out_dict["hybridization_true"],
        }

        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        edges_pred = out_dict["bonds_pred"]
        if self.hparams.addNumHs and self.hparams.addHybridization:
            a, b, c, d = (
                self.model.num_atom_types,
                self.model.num_charge_classes,
                self.model.num_Hs,
                self.model.num_hybridization,
            )
            atoms_pred, charges_pred, numHs_pred, hybridization_pred = atoms_pred.split(
                [a, b, c, d], dim=-1
            )
        elif self.hparams.addNumHs and not self.hparams.addHybridization:
            a, b, c = (
                self.model.num_atom_types,
                self.model.num_charge_classes,
                self.model.num_Hs,
            )
            atoms_pred, charges_pred, numHs_pred = atoms_pred.split([a, b, c], dim=-1)
            hybridization_pred = None
        elif self.hparams.addHybridization and not self.hparams.addNumHs:
            a, b, c = (
                self.model.num_atom_types,
                self.model.num_charge_classes,
                self.model.num_hybridization,
            )
            atoms_pred, charges_pred, hybridization_pred = atoms_pred.split(
                [a, b, c], dim=-1
            )
            numHs_pred = None
        else:
            a, b = self.model.num_atom_types, self.model.num_charge_classes
            atoms_pred, charges_pred = atoms_pred.split([a, b], dim=-1)
            numHs_pred = hybridization_pred = None

        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
            "numHs": numHs_pred,
            "hybridization": hybridization_pred,
        }

        true_data["properties"] = None
        pred_data["properties"] = None
        loss = self.diffusion_loss(
            true_data=true_data,
            pred_data=pred_data,
            batch=batch.pos_batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            intermediate_coords=self.hparams.store_intermediate_coords
            and self.training,
            weights=weights,
            node_level_t=self.node_level_t,
            variable_mask=out_dict["variable_mask"],
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
        )
        if self.hparams.addNumHs:
            final_loss += loss["numHs"]
        else:
            loss["numHs"] = None

        if self.hparams.addHybridization:
            final_loss += loss["hybridization"]
        else:
            loss["hybridization"] = None

        if self.hparams.use_latent_encoder:
            prior_loss = self.latent_loss(inputdict=out_dict.get("latent"))
            num_nodes_loss = F.cross_entropy(
                out_dict["latent"]["num_nodes_pred"],
                out_dict["latent"]["num_nodes_true"],
            )
            final_loss = (
                final_loss + self.hparams.prior_beta * prior_loss + num_nodes_loss
            )
        else:
            prior_loss = num_nodes_loss = None

        if torch.any(final_loss.isnan()):
            final_loss = final_loss[~final_loss.isnan()]
            print(f"Detected NaNs. Terminating training at epoch {self.current_epoch}")
            exit()

        if self.hparams.ligand_pocket_distance_loss:
            dloss0 = ligand_pocket_clash_energy_loss(
                pos_ligand=coords_pred,
                pos_pocket=out_dict["coords_pocket"],
                x_ligand=atoms_pred.argmax(dim=-1).detach(),
                x_pocket=out_dict["atoms_pocket"],
                batch_ligand=batch.batch,
                batch_pocket=batch.pos_pocket_batch,
                b=0.25,
                a=1.0,
            )

            # dloss1 = lddt_loss(
            #    pos_ligand_true=out_dict["coords_true"],
            #    pos_ligand_pred=coords_pred,
            #    pos_pocket=out_dict["coords_pocket"],
            #    batch_ligand=batch.batch,
            #    batch_pocket=batch.pos_pocket_batch,
            #    cutoff=10.0,
            # )
            dloss1 = 0.0
            dloss = 0.1 * (dloss0 + dloss1)
            if self.hparams.node_level_t:
                if weights is not None:
                    dloss = weights * dloss
                    dloss = dloss * out_dict["variable_mask"]
                    dloss = scatter_add(dloss, batch.batch, dim=0)
                    dloss = dloss.mean()
            else:
                if weights is not None:
                    dloss = scatter_mean(dloss, batch.batch, dim=0)
                    dloss = weights * dloss
                    dloss = dloss.sum()
            final_loss += dloss
        else:
            dloss = None

        self._log(
            final_loss,
            loss["coords"],
            loss["atoms"],
            loss["charges"],
            loss["numHs"],
            loss["bonds"],
            dloss,
            prior_loss,
            num_nodes_loss,
            loss["hybridization"],
            batch_size,
            stage,
        )

        return final_loss

    def _log(
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        numHs_loss,
        bonds_loss,
        d_loss,
        prior_loss,
        num_nodes_loss,
        hybridization_loss,
        batch_size,
        stage,
    ):
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=False,
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/charges_loss",
            charges_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        if numHs_loss is not None:
            assert self.hparams.addNumHs
            if self.hparams.addNumHs:
                self.log(
                    f"{stage}/numHs_loss",
                    numHs_loss,
                    on_step=True,
                    batch_size=batch_size,
                    prog_bar=(stage == "train"),
                    sync_dist=self.hparams.gpus > 1 and stage == "val",
                )
        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        if d_loss is not None:
            self.log(
                f"{stage}/d_loss",
                d_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if prior_loss is not None:
            self.log(
                f"{stage}/prior_loss",
                prior_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if num_nodes_loss is not None:
            self.log(
                f"{stage}/num_nodes_loss",
                num_nodes_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if hybridization_loss is not None:
            self.log(
                f"{stage}/hybridization_loss",
                hybridization_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )

    def configure_optimizers(self):

        all_params = list(self.model.parameters())

        if self.hparams.use_latent_encoder:
            all_params += list(self.latent_net.parameters())

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.hparams["lr"],
                amsgrad=True,
                weight_decay=1.0e-12,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                all_params,
                lr=self.hparams["lr"],
                momentum=0.9,
                nesterov=True,
            )
        if self.hparams["lr_scheduler"] == "reduce_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=self.hparams["lr_patience"],
                cooldown=self.hparams["lr_cooldown"],
                factor=self.hparams["lr_factor"],
            )
        elif self.hparams["lr_scheduler"] == "cyclic":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams["lr_min"],
                max_lr=self.hparams["lr"],
                mode="exp_range",
                step_size_up=self.hparams["lr_step_size"],
                cycle_momentum=False,
            )
        elif self.hparams["lr_scheduler"] == "one_cyclic":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams["lr"],
                steps_per_epoch=len(self.trainer.datamodule.train_dataset),
                epochs=self.hparams["num_epochs"],
            )
        elif self.hparams["lr_scheduler"] == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["lr_patience"],
                eta_min=self.hparams["lr_min"],
            )
        elif self.hparams["lr_scheduler"] == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.997
            )
        else:
            raise Exception("Scheduler not found")
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": self.validity,
            "strict": False,
        }
        return [optimizer], [scheduler]


###########################################################


class TrainerFlow(pl.LightningModule):

    def __init__(
        self,
        hparams: Dict[str, Any],
        dataset_info: DatasetInfo,
        ckpt_path: Optional[str] = None,
        pocket_noise_std: float = 0.1,
        smiles_train: Optional[List[str]] = None,
    ):

        super().__init__()

        if "model_type" not in hparams.keys():
            hparams["model_type"] = "protein-ligand"
        if "addNumHs" not in hparams.keys():
            hparams["addNumHs"] = False
        if "model" not in hparams.keys():
            hparams["model"] = "flow"
        if "fragmentation" not in hparams.keys():
            hparams["fragmentation"] = False
        if "fragmentation_mix" not in hparams.keys():
            hparams["fragmentation_mix"] = False
        if "fragment_prior" not in hparams.keys():
            hparams["fragment_prior"] = "fragment"  # redundant
        if "model_score" not in hparams.keys():
            hparams["model_score"] = False
        if "omit_intermediate_noise" not in hparams.keys():
            hparams["omit_intermediate_noise"] = False
        if "gamma2_func" in hparams.keys():
            hparams["gamma2_func"] = "t(1-t)"
        if "t_cond_frac" not in hparams.keys():
            hparams["t_cond_frac"] = 0.5

        self.save_hyperparameters(hparams)
        self.dataset_info = dataset_info

        if hparams["model_type"] == "ligand":
            print("Using Ligand Flow Model")
            self.model = EQGATDiffLFlow(hparams, dataset_info, ckpt_path)
        elif hparams["model_type"] == "protein-ligand":
            print("Using Protein-Ligand Flow Model")
            self.model = EQGATDiffPLFlow(hparams, dataset_info, ckpt_path)
        else:
            raise ValueError(
                "Invalid model type. Only ligand or protein-ligand supported"
            )

        self.hparams.cat_prior = "simplex-disc"
        assert (
            not self.hparams.use_latent_encoder
        ), "Latent encoder not supported in flow model"
        self.latent_net = None  # type: ignore [assignment]
        self.latent_loss = None  # type: ignore [assignment]

        modalities = ["coords", "atoms", "charges", "bonds"]
        if self.hparams.addNumHs:
            modalities.append("numHs")
        if self.hparams.addHybridization:
            modalities.append("hybridization")
        self.diffusion_loss = DiffusionLoss(
            modalities=modalities,
            param=["data"] * len(modalities),
        )
        self.pocket_noise_std = pocket_noise_std
        if "node_level_t" in hparams.keys():
            self.node_level_t = hparams["node_level_t"]
        else:
            self.node_level_t = False

        self.i = 0
        self.validity = 0.0
        self.connected_components = 0.0
        self.angles_w1 = 1000.0
        self.smiles_train = smiles_train

        self.validity_cond = 0.0
        self.connected_components_cond = 0.0
        self.angles_w1_cond = 1000.0

        self.time_sampler = UniformTimeSampler()
        print(self.time_sampler)

    def forward(
        self,
        batch: Batch,
        t: Tensor,
        latent_gamma: float = 1.0,
        fragment_anchor_mask: Optional[Tensor] = None,
    ):

        if self.latent_net is not None:
            if self.node_level_t:
                raise ValueError("Node level t not supported with latent encoder")
            latent_out = self.latent_net(batch, t)
            z = latent_out["z_true"]
        else:
            latent_out = None
            z = None

        out = self.model(
            batch=batch,
            t=t,
            z=z,
            latent_gamma=latent_gamma,
            pocket_noise_std=self.pocket_noise_std,
            # defaults to 0.1 - check in future to remove.
            fragment_anchor_mask=fragment_anchor_mask,
        )

        if latent_out is not None:
            out["latent"] = latent_out

        return out

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        log = False
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:
            if self.local_rank == 0:
                print(f"Running evaluation in epoch {self.current_epoch + 1}")

            if self.local_rank == 0:
                print("Running unconditional evaluation")

            final_res = self.run_evaluation_unconditional(
                step=self.i,
                verbose=True,
                inner_verbose=False,
                device="cuda",
                prior_n_atoms="conditional",
            )
            if log:
                self.log(
                    name="val/validity",
                    value=final_res.validity.tolist()[0],
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    name="val/uniqueness",
                    value=final_res.uniqueness.tolist()[0],
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    name="val/novelty",
                    value=final_res.novelty.tolist()[0],
                    on_epoch=True,
                    sync_dist=True,
                )

            if self.hparams.fragmentation:
                if self.local_rank == 0:
                    print("Running conditional evaluation")
                final_res = self.run_evaluation_conditional(
                    step=self.i,
                    verbose=True,
                    device="cuda",
                )
                if log:
                    self.log(
                        name="val/validity_cond",
                        value=final_res.validity.tolist()[0],
                        on_epoch=True,
                        sync_dist=True,
                    )
                    self.log(
                        name="val/uniqueness_cond",
                        value=final_res.uniqueness.tolist()[0],
                        on_epoch=True,
                        sync_dist=True,
                    )
                    self.log(
                        name="val/novelty_cond",
                        value=final_res.novelty.tolist()[0],
                        on_epoch=True,
                        sync_dist=True,
                    )

            # make sure always save on validation
            save_path = os.path.join(
                self.hparams.save_dir, f"ckpt_epoch_{str(self.current_epoch + 1)}.ckpt"
            )
            self.trainer.save_checkpoint(save_path)

            self.i += 1

    @torch.no_grad()
    # TODO: refactor function signature
    def run_evaluation_unconditional(
        self,
        step: int,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        inner_verbose=False,
        save_traj=False,
        use_ligand_dataset_sizes: bool = False,
        run_test_eval: bool = False,
        n_nodes_bias: int = 0,
        device: str = "cpu",
        prior_n_atoms: str = "reference",
    ):
        """
        Runs the evaluation on the entire validation dataloader.
        Generates 1 ligand in 1 receptor structure
        """

        dataloader = (
            self.trainer.datamodule.val_dataloader()
            if not run_test_eval
            else self.trainer.datamodule.test_dataloader()
        )
        molecule_list = []
        start = datetime.now()

        for i, data in enumerate(dataloader):

            num_graphs = len(data.batch.bincount())
            if use_ligand_dataset_sizes or prior_n_atoms == "reference":
                num_nodes_lig = data.batch.bincount().to(self.device)
            elif prior_n_atoms == "conditional":
                if self.hparams.model_type == "protein-ligand":
                    num_nodes_lig = self.dataset_info.conditional_size_distribution.sample_conditional(  # noqa: B950
                        n1=None, n2=data.pos_pocket_batch.bincount()
                    ).to(
                        self.device
                    )
                else:
                    num_nodes_lig = self.dataset_info.nodes_dist.sample_n(
                        n_samples=num_graphs, device=self.device
                    ).to(self.device)
                num_nodes_lig += n_nodes_bias
            else:
                raise ValueError("Invalid prior_n_atoms")

            if self.hparams.use_latent_encoder:
                _ = self.latent_net(data.to(self.device), t=None)["z_true"]
            else:
                _ = None

            molecules, _, cl_loss = self.model.reverse_sampling(
                N=100,
                num_graphs=num_graphs,
                device=self.device,
                num_nodes_lig=num_nodes_lig,
                pocket_data=data,
                verbose=inner_verbose,
                save_traj=save_traj,
                cat_noise=True,
                pos_noise=False,
                score_dynamics=self.hparams.model_score,
                score_scale=0.5,
                discrete_gat=False,
            )
            molecule_list.extend(molecules)
            del data
            torch.cuda.empty_cache()

        # evaluate all generated samples from the validation set
        try:
            outs = analyze_stability_for_molecules(
                dataset_info=self.dataset_info,
                device=device,
                smiles_train=self.smiles_train,
                molecule_list=molecule_list,
                remove_hs=self.hparams.remove_hs,
                local_rank=self.local_rank,
                test=run_test_eval,
                filter_by_lipinski=False,
                filter_by_posebusters=False,
                pdb_file=None,
            )
        except Exception as e:
            print(f"Error in analyze_stability_for_molecules: {e}")
            outs = {
                "validity_dict": {"validity": 0.0},
                "statistics_dict": {
                    "connected_components": 0.0,
                    "sampling/AnglesW1": 1000.0,
                },
                "stability_dict": {"stability": 0.0},
            }
            cl_loss = -1.0

        if not run_test_eval:
            save_cond = (
                self.validity < outs["validity_dict"]["validity"]
                and self.connected_components
                <= outs["statistics_dict"]["connected_components"]
            )
            save_cond = (
                save_cond
                or outs["statistics_dict"]["sampling/AnglesW1"] < self.angles_w1
            )
        else:
            save_cond = False

        if save_cond:
            self.validity = outs["validity_dict"]["validity"]
            self.connected_components = outs["statistics_dict"]["connected_components"]
            self.angles_w1 = outs["statistics_dict"]["sampling/AnglesW1"]
            save_path = os.path.join(self.hparams.save_dir, "best_valid.ckpt")
            self.trainer.save_checkpoint(save_path)

        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")
        total_res_dict = dict(outs["stability_dict"])
        total_res_dict.update(outs["validity_dict"])
        total_res_dict.update(outs["statistics_dict"])
        total_res_dict.update({"clash_loss": cl_loss})

        total_res: pd.DataFrame = pd.DataFrame.from_dict([total_res_dict])
        if self.local_rank == 0:
            print(total_res)
        total_res["step"] = str(step)
        total_res["epoch"] = str(self.current_epoch)
        total_res["run_time"] = str(run_time)
        if save_dir is None:
            save_dir = os.path.join(
                self.hparams.save_dir,
                "run" + str(self.hparams.id),
                "evaluation.csv",
            )
        else:
            save_dir = os.path.join(save_dir, "evaluation.csv")
        if self.local_rank == 0:
            print(f"Saving evaluation csv file to {save_dir}")
            with open(save_dir, "a") as f:
                total_res.to_csv(f, header=True)

        return total_res

    def run_evaluation_conditional(
        self,
        step: int,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        run_test_eval: bool = False,
        device: str = "cpu",
    ):
        """
        Runs the evaluation on the entire validation dataloader.
        Generates 1 ligand in 1 receptor structure
        """

        dataloader = (
            self.trainer.datamodule.val_dataloader()
            if not run_test_eval
            else self.trainer.datamodule.test_dataloader()
        )
        molecule_list = []
        start = datetime.now()

        # make sure the following attributes are there
        # .data_mask, .lig_inpaint_mask, .anchor_mask, .n_variable

        for _, data in enumerate(dataloader):
            data = data.to(self.device)
            batch_num_nodes = data.batch.bincount()
            fragment_anchor_mask = (
                torch.from_numpy(
                    np.concatenate(
                        [
                            get_random_fragment_anchor_mask(
                                frag_ids, anchor_ids, n.item()
                            )
                            for frag_ids, anchor_ids, n in zip(
                                data.sub_ids, data.anchor_ids, batch_num_nodes
                            )
                        ]
                    )
                )
                .to(data.x.device)
                .float()
            )
            n = data.x.shape[0]
            data.data_mask = torch.ones((n,), device=data.x.device, dtype=torch.bool)
            data.lig_inpaint_mask = ~(fragment_anchor_mask[:, 0].bool())  # kept fixed
            data.anchor_mask = fragment_anchor_mask[:, 1]
            data.n_variable = scatter_add(
                fragment_anchor_mask[:, 0],
                data.batch,
                dim=0,
                dim_size=len(batch_num_nodes),
            )
            molecules, _, cl_loss = self.model.reverse_sampling_node_level_t(
                N=100,
                device=data.x.device,
                pocket_data=data,
                verbose=False,
                save_traj=False,
                cat_noise=True,
                pos_noise=False,
                clash_guidance=False,
                clash_guidance_scale=0.02,
                score_dynamics=self.hparams.model_score,
                score_scale=0.5,
                dt_pow=0.5,
                pos_context_noise=0.0,
                discrete_gat=False,
            )

            molecule_list.extend(molecules)
            del data
            torch.cuda.empty_cache()

        # evaluate all generated samples from the validation set
        try:
            outs = analyze_stability_for_molecules(
                dataset_info=self.dataset_info,
                device=device,
                smiles_train=self.smiles_train,
                molecule_list=molecule_list,
                remove_hs=self.hparams.remove_hs,
                local_rank=self.local_rank,
                test=run_test_eval,
                filter_by_lipinski=False,
                filter_by_posebusters=False,
                pdb_file=None,
            )
        except Exception as e:
            print(f"Error in analyze_stability_for_molecules: {e}")
            outs = {
                "validity_dict": {"validity": 0.0},
                "statistics_dict": {
                    "connected_components": 0.0,
                    "sampling/AnglesW1": 1000.0,
                },
                "stability_dict": {"stability": 0.0},
            }
            cl_loss = -1.0

        if not run_test_eval:
            save_cond = (
                self.validity_cond < outs["validity_dict"]["validity"]
                and self.connected_components_cond
                <= outs["statistics_dict"]["connected_components"]
            )
            save_cond = (
                save_cond
                or outs["statistics_dict"]["sampling/AnglesW1"] < self.angles_w1_cond
            )
        else:
            save_cond = False

        if save_cond:
            self.validity_cond = outs["validity_dict"]["validity"]
            self.connected_components_cond = outs["statistics_dict"][
                "connected_components"
            ]
            self.angles_w1_cond = outs["statistics_dict"]["sampling/AnglesW1"]
            save_path = os.path.join(self.hparams.save_dir, "best_valid_cond.ckpt")
            self.trainer.save_checkpoint(save_path)

        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")
        total_res_dict = dict(outs["stability_dict"])
        total_res_dict.update(outs["validity_dict"])
        total_res_dict.update(outs["statistics_dict"])
        total_res_dict.update({"clash_loss": cl_loss})

        total_res: pd.DataFrame = pd.DataFrame.from_dict([total_res_dict])
        if self.local_rank == 0:
            print(total_res)
        total_res["step"] = str(step)
        total_res["epoch"] = str(self.current_epoch)
        total_res["run_time"] = str(run_time)
        if save_dir is None:
            save_dir = os.path.join(
                self.hparams.save_dir,
                "run" + str(self.hparams.id),
                "evaluation_conditional.csv",
            )
        else:
            save_dir = os.path.join(save_dir, "evaluation_conditional.csv")
        if self.local_rank == 0:
            with open(save_dir, "a") as f:
                total_res.to_csv(f, header=True)
            print(f"Saving evaluation csv file to {save_dir}")

        return total_res

    def step_fnc(self, batch, batch_idx, stage: str):

        batch.batch = batch.pos_batch
        batch_size = int(batch.batch.max()) + 1
        batch_num_nodes = batch.batch.bincount()
        t = self.time_sampler.sample((batch_size, 1)).to(batch.x.device)
        if self.node_level_t:
            N = batch.x.size(0)
            t_batch = t.repeat_interleave(batch_num_nodes)
            if not self.hparams.fragmentation:
                # just mix all node levels
                mask_batch = (
                    torch.rand((batch_size, 1), device=batch.x.device)
                    < (1.0 - self.hparams.t_cond_frac)
                ).repeat_interleave(batch_num_nodes)
                t_node = torch.rand(size=(N, 1), device=batch.x.device)
                t = t_batch * mask_batch.float() + t_node * (1 - mask_batch.float())
                t_batch = t_batch.unsqueeze(-1)
                t = t_batch * mask_batch.float() + t_node * (1 - mask_batch.float())
                fragment_anchor_mask = None
            else:
                # conditional models

                fragment_anchor_mask = (
                    torch.from_numpy(
                        np.concatenate(
                            [
                                get_random_fragment_anchor_mask(
                                    frag_ids, anchor_ids, n.item()
                                )
                                for frag_ids, anchor_ids, n in zip(
                                    batch.sub_ids, batch.anchor_ids, batch_num_nodes
                                )
                            ]
                        )
                    )
                    .to(batch.x.device)
                    .float()
                )

                if self.hparams.fragmentation_mix:
                    t_variable = (
                        torch.rand((batch_size, 1), device=batch.x.device)
                        < (1.0 - self.hparams.t_cond_frac)
                    ).repeat_interleave(batch_num_nodes)
                    # 0 or 1
                    t_variable = t_variable.float()
                    t_context = 1.0 - t_variable
                    ones = torch.ones_like(t_batch)
                    t = t_batch * (
                        t_variable * ones.float()
                        + t_context * fragment_anchor_mask[:, 0].float()
                    )
                    t_ = torch.empty_like(fragment_anchor_mask)
                    t_[:, 0] = 1  # fragment 1 means all is variable, 0 means fixed
                    t_[:, 1] = 0  # anchor 0 shows no anchor
                    fragment_anchor_mask = (
                        t_context.unsqueeze(-1) * fragment_anchor_mask.float()
                        + t_variable.unsqueeze(-1) * t_.float()
                    )
                else:
                    if t_batch.ndim == 1:
                        t_batch = t_batch.unsqueeze(-1)
                    t = t_batch * fragment_anchor_mask[:, 0].float().unsqueeze(-1)

                t[t == 0] = 1.0  # data state
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
        else:
            fragment_anchor_mask = None

        s = self.model.cat_atoms.scheduler(t)

        if isinstance(self.time_sampler, UniformTimeSampler):
            weights = s.alpha_t / s.sigma_t
            weights = torch.clamp(weights, min=0.05, max=1.5).squeeze(1)
        else:
            weights = torch.ones_like(s.alpha_t)

        out_dict = self(
            batch=batch,
            t=t,
            latent_gamma=1.0,
            fragment_anchor_mask=fragment_anchor_mask,
        )
        true_data = {
            "coords": out_dict["coords_true"],
            "atoms": out_dict["atoms_true"],
            "charges": out_dict["charges_true"],
            "bonds": out_dict["bonds_true"],
            "numHs": out_dict["numHs_true"],
            "hybridization": out_dict["hybridization_true"],
        }

        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        edges_pred = out_dict["bonds_pred"]

        if self.hparams.addNumHs and self.hparams.addHybridization:
            a, b, c, d = (
                self.model.num_atom_types,
                self.model.num_charge_classes,
                self.model.num_Hs,
                self.model.num_hybridization,
            )
            atoms_pred, charges_pred, numHs_pred, hybridization_pred = atoms_pred.split(
                [a, b, c, d], dim=-1
            )
        elif self.hparams.addNumHs and not self.hparams.addHybridization:
            a, b, c = (
                self.model.num_atom_types,
                self.model.num_charge_classes,
                self.model.num_Hs,
            )
            atoms_pred, charges_pred, numHs_pred = atoms_pred.split([a, b, c], dim=-1)
            hybridization_pred = None
        elif self.hparams.addHybridization and not self.hparams.addNumHs:
            a, b, c = (
                self.model.num_atom_types,
                self.model.num_charge_classes,
                self.model.num_hybridization,
            )
            atoms_pred, charges_pred, hybridization_pred = atoms_pred.split(
                [a, b, c], dim=-1
            )
            numHs_pred = None
        else:
            a, b = self.model.num_atom_types, self.model.num_charge_classes
            atoms_pred, charges_pred = atoms_pred.split([a, b], dim=-1)
            numHs_pred = hybridization_pred = None

        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
            "numHs": numHs_pred,
            "hybridization": hybridization_pred,
        }

        true_data["properties"] = None
        pred_data["properties"] = None

        loss = self.diffusion_loss(
            true_data=true_data,
            pred_data=pred_data,
            batch=batch.pos_batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            intermediate_coords=self.hparams.store_intermediate_coords
            and self.training,
            weights=weights,
            node_level_t=self.node_level_t,
            variable_mask=out_dict["variable_mask"],
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
        )

        if self.hparams.addNumHs:
            final_loss += loss["numHs"]
        else:
            loss["numHs"] = None

        if self.hparams.addHybridization:
            final_loss += loss["hybridization"]
        else:
            loss["hybridization"] = None

        if self.hparams.use_latent_encoder:
            prior_loss = self.latent_loss(inputdict=out_dict.get("latent"))
            num_nodes_loss = F.cross_entropy(
                out_dict["latent"]["num_nodes_pred"],
                out_dict["latent"]["num_nodes_true"],
            )
            final_loss = (
                final_loss + self.hparams.prior_beta * prior_loss + num_nodes_loss
            )
        else:
            prior_loss = num_nodes_loss = None

        if torch.any(final_loss.isnan()):
            final_loss = final_loss[~final_loss.isnan()]
            print(f"Detected NaNs. Terminating training at epoch {self.current_epoch}")
            exit()

        if self.hparams.ligand_pocket_distance_loss:
            dloss0 = ligand_pocket_clash_energy_loss(
                pos_ligand=coords_pred,
                pos_pocket=out_dict["coords_pocket"],
                x_ligand=atoms_pred.argmax(dim=-1).detach(),
                x_pocket=out_dict["atoms_pocket"],
                batch_ligand=batch.batch,
                batch_pocket=batch.pos_pocket_batch,
                b=0.25,
                a=1.0,
                min_distance=2.0,
            )

            # dloss1 = lddt_loss(
            #    pos_ligand_true=out_dict["coords_true"],
            #    pos_ligand_pred=coords_pred,
            #    pos_pocket=out_dict["coords_pocket"],
            #    batch_ligand=batch.batch,
            #    batch_pocket=batch.pos_pocket_batch,
            #    cutoff=10.0,
            # )
            dloss1 = 0.0
            dloss = 0.05 * (dloss0 + dloss1)
            if self.hparams.node_level_t:
                if weights is not None:
                    dloss = weights * dloss
                    dloss = dloss * out_dict["variable_mask"]
                    dloss = scatter_add(dloss, batch.batch, dim=0)
                    dloss = dloss.mean()
            else:
                if weights is not None:
                    dloss = scatter_mean(dloss, batch.batch, dim=0)
                    dloss = weights * dloss
                    dloss = dloss.sum()
            final_loss += dloss
        else:
            dloss = None

        if self.hparams.model_score:
            eps = out_dict["eps"]
            sigma = out_dict["sigma"]
            # score = 2.0 * sigma * out_dict["score_pred"]
            score = sigma * out_dict["score_pred"]
            score_loss = (score + eps).pow(2).sum(-1)
            score_loss = score_loss.mean()
            final_loss += score_loss
        else:
            score_loss = None

        self._log(
            final_loss,
            loss["coords"],
            loss["atoms"],
            loss["charges"],
            loss["numHs"],
            loss["bonds"],
            score_loss,
            dloss,
            prior_loss,
            num_nodes_loss,
            loss["hybridization"],
            batch_size,
            stage,
        )

        return final_loss

    def _log(
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        numHs_loss,
        bonds_loss,
        score_loss,
        d_loss,
        prior_loss,
        num_nodes_loss,
        hybridization_loss,
        batch_size,
        stage,
    ):
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=False,
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/charges_loss",
            charges_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        if numHs_loss is not None:
            assert self.hparams.addNumHs
            if self.hparams.addNumHs:
                self.log(
                    f"{stage}/numHs_loss",
                    numHs_loss,
                    on_step=True,
                    batch_size=batch_size,
                    prog_bar=(stage == "train"),
                    sync_dist=self.hparams.gpus > 1 and stage == "val",
                )
        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        if d_loss is not None:
            self.log(
                f"{stage}/d_loss",
                d_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if score_loss is not None:
            self.log(
                f"{stage}/score_loss",
                score_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if prior_loss is not None:
            self.log(
                f"{stage}/prior_loss",
                prior_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if num_nodes_loss is not None:
            self.log(
                f"{stage}/num_nodes_loss",
                num_nodes_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
        if hybridization_loss is not None:
            self.log(
                f"{stage}/hybridization_loss",
                hybridization_loss,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )

    def configure_optimizers(self):

        all_params = list(self.model.parameters())

        if self.hparams.use_latent_encoder:
            all_params += list(self.latent_net.parameters())

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.hparams["lr"],
                amsgrad=True,
                weight_decay=1.0e-12,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                all_params,
                lr=self.hparams["lr"],
                momentum=0.9,
                nesterov=True,
            )
        if self.hparams["lr_scheduler"] == "reduce_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=self.hparams["lr_patience"],
                cooldown=self.hparams["lr_cooldown"],
                factor=self.hparams["lr_factor"],
            )
        elif self.hparams["lr_scheduler"] == "cyclic":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams["lr_min"],
                max_lr=self.hparams["lr"],
                mode="exp_range",
                step_size_up=self.hparams["lr_step_size"],
                cycle_momentum=False,
            )
        elif self.hparams["lr_scheduler"] == "one_cyclic":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams["lr"],
                steps_per_epoch=len(self.trainer.datamodule.train_dataset),
                epochs=self.hparams["num_epochs"],
            )
        elif self.hparams["lr_scheduler"] == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["lr_patience"],
                eta_min=self.hparams["lr_min"],
            )
        elif self.hparams["lr_scheduler"] == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.997
            )
        else:
            raise Exception("Scheduler not found")
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": self.validity,
            "strict": False,
        }
        return [optimizer], [scheduler]

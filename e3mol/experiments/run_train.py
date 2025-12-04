import os
import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment

from e3mol.experiments.callbacks import (  # type: ignore # noqa: F401
    ExponentialMovingAverage,
)
from e3mol.experiments.data.datainfo import load_dataset_info
from e3mol.experiments.data.fragmentation import FragmentTransform
from e3mol.experiments.hparams import add_arguments
from e3mol.experiments.trainer import Trainer, TrainerFlow

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

if __name__ == "__main__":

    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    if not os.path.isdir(hparams.save_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.save_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")
    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + f"/run{hparams.id}/",
        save_top_k=3,
        monitor="val/loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
    )
    print(f"Loading {hparams.dataset} Datamodule.")
    if hparams.dataset in ["crossdocked", "kinodata"]:
        from e3mol.experiments.data.dataset import LigandPocketDataModule as DataModule
    else:
        raise ValueError(f"Unknown dataset {hparams.dataset}")

    if hparams.fragmentation:
        assert hparams.node_level_t, "Node level timestep is required for fragmentation"
        transform = FragmentTransform(
            minFragmentSize=hparams.minimum_fragment_size,
            method=hparams.fragmentation_method,
            include_middle_fragment=hparams.include_middle_fragment,
            cuttable_pattern=hparams.cuttable_pattern,
        )
        validation_transform = FragmentTransform(
            minFragmentSize=hparams.minimum_fragment_size,
            method="briccs",
        )
    else:
        transform = None
        validation_transform = None

    datamodule = DataModule(
        hparams, transform=transform, validation_transform=validation_transform
    )
    datamodule.setup()
    statistics_dict_path = hparams.dataset_root + "/processed/all_stats_dict_noh.pickle"
    dataset_info = load_dataset_info(
        name=hparams.dataset,
        statistics_dict_path=statistics_dict_path,
        ligand_pocket_histogram_path=None,
        dataset=hparams.dataset,
    )

    if hparams.load_ckpt_from_pretrained is not None:
        assert hparams.load_ckpt is None

    if hparams.load_ckpt is not None:
        assert hparams.load_ckpt_from_pretrained is None

    if hparams.model == "diffusion":
        model = Trainer(
            hparams=hparams.__dict__,
            dataset_info=dataset_info,
            pocket_noise_std=hparams.pocket_noise_std,
            ckpt_path=hparams.load_ckpt_from_pretrained,
            smiles_train=datamodule.train_dataset.smiles,
        )
    else:
        model = TrainerFlow(
            hparams=hparams.__dict__,
            dataset_info=dataset_info,
            pocket_noise_std=hparams.pocket_noise_std,
            ckpt_path=hparams.load_ckpt_from_pretrained,
            smiles_train=datamodule.train_dataset.smiles,
        )

    strategy = "ddp" if hparams.gpus > 1 else "auto"
    callbacks = [
        ema_callback,
        lr_logger,
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
    ]

    if hparams.ema_decay == 1.0:
        callbacks = callbacks[1:]

    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else 1,
        strategy=strategy,
        plugins=LightningEnvironment(),
        num_nodes=1,
        logger=tb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=callbacks,
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
    )

    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt,
    )

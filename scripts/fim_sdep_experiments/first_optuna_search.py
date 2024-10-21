"""
Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch Lightning, and FashionMNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole FashionMNIST dataset, we here use a small subset of it.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python pytorch_lightning_simple.py [--pruning]

"""

import argparse
import os
from typing import List
from typing import Optional

import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms


if version.parse(pl.__version__) < version.parse("1.6.0"):
    raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()

from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams
from fimodemix.data.dataloaders import (
    FIMSDEpDataModule
)
from fimodemix.utils.experiment_files import ExperimentsFiles
from fimodemix.models.fim_sdep import FIMSDEp
from fimodemix.pipelines.sdep_pipeline import FIMSDEpPipeline

EPOCHS = 2

def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    #dim_time = trial.suggest_int("dim_time", 19, 20)
    dim_time = 19
    x0_hidden_layers = [
        trial.suggest_int("x0_hidden_layers_l{}".format(i), 50, 100, log=True) for i in range(2)
    ]

    # Set Parameters
    params = FIMSDEpModelParams(dim_time=dim_time,
                                x0_hidden_layers=x0_hidden_layers,
                                num_epochs=EPOCHS)
    # Set up Dataloaders
    dataloaders = FIMSDEpDataModule(params)
    # Set up Model
    model = FIMSDEp(params)

    #model = LightningNet(dropout, output_dims)
    #datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, 
                                                   monitor="val_loss")],
    )
    hyperparameters = dict(dim_time=dim_time, x0_hidden_layers=x0_hidden_layers)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model,datamodule=dataloaders)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
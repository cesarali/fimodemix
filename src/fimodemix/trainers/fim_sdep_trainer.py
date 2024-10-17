
import os
import time

import torch
import shutil
import numpy as np
import torch.nn as nn

from pathlib import Path
import pytorch_lightning as pl
from dataclasses import dataclass
from dataclasses import dataclass,asdict, field
from typing import Any, Dict, Optional, Union, List,Tuple

from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams
from fimodemix.data.dataloaders import (
    FIMSDEpDataLoader
)
from fimodemix.utils.experiment_files import ExperimentsFiles
from fimodemix.models.fim_sde import FIMSDEp

def train_fim_sde_p(params:FIMSDEpModelParams):
    """
    This function creates an MLFlow Logger and a Lightning Trainer to train FIMSDE_p
    """
    # Experiment Files
    experiment_files = ExperimentsFiles(experiment_indentifier=None,
                                        delete=True)
    # Set up TensorBoard logger
    # logger = MLFlowLogger(experiment_name="time_series_transformer",tracking_uri='http://localhost:5000')
    
    logger = TensorBoardLogger(experiment_files.tensorboard_dir, 
                               name=experiment_files.experiment_indentifier)

    # Set up Model Checkpointing
    checkpoint_callback_best = ModelCheckpoint(dirpath=experiment_files.checkpoints_dir,
                                               save_top_k=1, 
                                               monitor="val_loss",
                                               filename="best-{epoch:02d}")
    
    checkpoint_callback_last = ModelCheckpoint(dirpath=experiment_files.checkpoints_dir,
                                               save_top_k=1,
                                               monitor=None,
                                               filename="last-{epoch:02d}")
    
    # Set up Dataloaders
    dataloaders = FIMSDEpDataLoader(params)

    # Set up Model
    model = FIMSDEp(params)
    model.save_hyperparameters_to_yaml(experiment_files.params_yaml)

    #Set up trainers
    trainer = Trainer(
        default_root_dir=experiment_files.experiment_dir,
        logger=logger,
        max_epochs=params.num_epochs,
        callbacks=[checkpoint_callback_best,
                   checkpoint_callback_last]
    )

    trainer.fit(model, 
                dataloaders.train_it,
                dataloaders.validation_it)

if __name__=="__main__":
    params = FIMSDEpModelParams()
    train_fim_sde_p(params)


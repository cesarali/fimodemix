import os
import torch

from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams
from fimodemix.data.dataloaders import (
    FIMSDEpDataLoader
)

def test_dataloader_p():
    # Set Parameters
    params = FIMSDEpModelParams()
    # Set up Dataloaders
    dataloaders = FIMSDEpDataLoader(params)
    print(dataloaders.one_batch.diffusion_parameters.shape)
    print(dataloaders.one_batch.hypercube_locations.shape)
    print(dataloaders.one_batch.obs_times.shape)
    print(dataloaders.one_batch.drift_at_hypercube.shape)
    
if __name__=="__main__":
    test_dataloader_p()

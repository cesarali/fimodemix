import os
import torch

from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams
from fimodemix.data.dataloaders import (
    FIMSDEpDataLoader
)
from fimodemix.models.fim_sde import FIMSDEp

def test_fim_sde_p():
    #Set Parameters
    params = FIMSDEpModelParams()
    #Set up Dataloaders
    dataloaders = FIMSDEpDataLoader(params)
    # Set up Model
    model = FIMSDEp(params)
    # call forward
    databatch = dataloaders.one_batch
    f_hats = model(databatch)

def test_fim_sde_p_loss():
    #Set Parameters
    params = FIMSDEpModelParams()
    #Set up Dataloaders
    dataloaders = FIMSDEpDataLoader(params)
    # Set up Model
    model = FIMSDEp(params)
    # call forward
    databatch = dataloaders.one_batch
    f_hats = model(databatch)
    loss = model.loss(f_hat=f_hats,
                      drift_at_hypercube=databatch.drift_at_hypercube,
                      mask=databatch.mask)
    print(loss)

if __name__=="__main__":
    test_fim_sde_p_loss()
    

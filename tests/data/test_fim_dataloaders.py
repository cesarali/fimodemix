import torch
from pprint import pprint
from dataclasses import dataclass,asdict
from fimodemix.configs.config_classes.fim_sde import FIMSDEModelParams
import numpy as np

from fimodemix.data.datasets import (
    FIMSDEDataset,
    FIMSDEDatabatch
)

if __name__=="__main__":
    # Define Params
    params = FIMSDEModelParams()
    # Define Data Set
    dataset = FIMSDEDataset(params=params)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=24, 
                                              shuffle=True)
    databatch = next(data_loader.__iter__())
    obs_values, obs_times, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask = databatch
    
    #for databatch in data_loader:
    #    print(len(databatch))




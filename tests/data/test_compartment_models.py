import os
import torch
import numpy as np

from fimodemix.data.generation_compartments import (
    OneCompartmentModel,
    OneCompartmentModelParams,
    define_compartment_models_from_yaml
)
from fimodemix.configs.config_classes.fim_compartments_config import FIMCompartmentModelParams
from fimodemix.data.datasets import FIMCompartmentDataset,FIMCompartementsDatabatchTuple
from fimodemix.data.dataloaders import FIMCompartmentDataloader
from torch.utils.data import DataLoader

def test_one_compartment():
    one_compartment_params = OneCompartmentModelParams(t_obs=[0.5, 1, 2, 4, 8, 16, 24])
    model = OneCompartmentModel(one_compartment_params)
    model_data = model.simulate()
    print(model_data.hidden_values.shape)

def test_read_all_models():
    compartments_hyperparameters_file = r"C:\Users\cesar\Desktop\Projects\FoundationModels\fimodemix\configs\compartment-generation\compartments_params.yaml"
    parameters = define_compartment_models_from_yaml(compartments_hyperparameters_file)

def test_compartment_dataset():
    one_compartment_params = OneCompartmentModelParams(t_obs=[0.5, 1, 2, 4, 8, 16, 24])
    model = OneCompartmentModel(one_compartment_params)
    model_data1 = model.simulate()
    print(model_data1.obs_times.shape)
    one_compartment_params = OneCompartmentModelParams(t_obs=[0.5, 1, 2, 4])
    model = OneCompartmentModel(one_compartment_params)
    model_data2 = model.simulate()
    print(model_data2.obs_times.shape)

    dataset = FIMCompartmentDataset(None,[model_data1,model_data2])
    dataloader = DataLoader(dataset,batch_size=32)
    databatch:FIMCompartementsDatabatchTuple = next(dataloader.__iter__())
    print(databatch.obs_values.shape)
    

def test_compartment_dataloaders():
    params = FIMCompartmentModelParams()
    dataloaders = FIMCompartmentDataloader(params)

if __name__=="__main__":
    test_compartment_dataset()
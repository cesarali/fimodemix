import os
import torch
import numpy as np

from fimodemix.data.generation_compartments import (
    OneCompartmentModel,
    OneCompartmentModelParams,
    define_compartment_models_from_yaml
)

def test_one_compartment():
    one_compartment_params = OneCompartmentModelParams(t_obs=[0.5, 1, 2, 4, 8, 16, 24])
    model = OneCompartmentModel(one_compartment_params)
    model_data = model.simulate()

def test_read_all_models():
    compartments_file = r"C:\Users\cesar\Desktop\Projects\FoundationModels\fimodemix\configs\compartment-generation\compartments_params.yaml"
    parameters = define_compartment_models_from_yaml(compartments_file)
    
if __name__=="__main__":
    test_read_all_models()
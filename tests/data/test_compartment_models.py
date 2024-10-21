import os
import torch

from fimodemix.data.generation_compartments import (
    OneCompartmentModel,
    OneCompartmentModelParams
)

def test_one_compartment():
    one_compartment_params = OneCompartmentModelParams()
    model = OneCompartmentModel(one_compartment_params)
    model_data = model.simulate()
    print(model_data.obs_values.shape)

if __name__=="__main__":
    test_one_compartment()
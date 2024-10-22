import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from fimodemix import data_path

@dataclass
class FIMCompartmentModelParams:
    # data ------------------------------------------------------------------
    data_name:str = "dummy_compartment_damped"
    data_path:str = str(data_path)
    compartments_hyperparameters_file:str = r"C:\Users\cesar\Desktop\Projects\FoundationModels\fimodemix\configs\compartment-generation\compartments_params.yaml"

    batch_size:int = 32
    test_batch_size:int = 32

    #experiments
    experiment_name:str = "test"
    
    # these values are updated after dataset is read ------------------------
    max_dimension: int = 1
    max_hypercube_size: int = 1
    max_num_steps: int = 1

    # Additional parameters -------------------------------------------------
    max_time_steps: int = 1
    max_dosing_steps: int = 1
    max_hidden_dimension: int = 1
    max_param_size: int = 1
    max_dosing_times: int = 1

    # model architecture --------------------------------------------------
    n_heads: int = 4 # used for all transformers
    dim_time:int = 19

    # phi_0 / first data encoding
    x0_hidden_layers: List[int] = field(default_factory=lambda:[50,50])
    x0_out_features: int = 21
    x0_dropout: float = 0.2

    encoding0_dim: int = 40
    #psi_1 / first transformer
    psi1_hidden_dim:int = 300
    psi1_nlayers:int = 2

    #Multiheaded Attention 1 / first path summary
    query_dim:int = 10

    hidden_dim: int = 64
    output_size: int = 1
    batch_size: int = 32
    seq_length: int = 10

    # training ---------------------------------------------------------------

    num_epochs: int = 10
    learning_rate: float = 0.001
    embed_dim: int = 8  # New embedding dimension

    # pipeline --------------------------------------------------------------

    dt_pipeline:float = 0.01
    number_of_time_steps_pipeline:int = 128

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FIMSDEpModelParams':
        with open(yaml_path, 'r') as file:
            params_dict = yaml.safe_load(file)
        return cls(**params_dict)
    
    def __post_init__(self):
        self.dim_time = self.dim_time*self.n_heads
        self.x0_out_features = self.x0_out_features*self.n_heads
        self.encoding0_dim = self.x0_out_features + self.dim_time 

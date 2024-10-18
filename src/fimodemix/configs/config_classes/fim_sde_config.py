import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from fimodemix import data_path

@dataclass
class FIMSDEpModelParams:
    # data ------------------------------------------------------------------
    data_name = "dummy_lorenz_damped"
    data_path = str(data_path)
    input_size: int = 1  # Original input size
    batch_size:int = 32
    test_batch_size:int = 32

    #experiments
    experiment_name:str = "test"
    
    # these values are updated after dataset is read
    max_dimension:int = 1
    max_hypercube_size:int = 1
    max_num_steps:int = 1

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

# PATRIK FUNCTIONALITY ------------------------------------------
@dataclass
class DataInFiles:
    obs_times: str
    obs_values: str
    locations: str
    drift_at_locations: str
    diffusion_at_locations: str

@dataclass
class DatasetPathCollections:
    train: List[str]
    test: List[str]

@dataclass
class PatrikModelConfig:    
    temporal_embedding_size: int = 256
    spatial_embedding_size: int = 256
    spatial_embedding_hidden_layers: Optional[List[int]] = field(default=None)
    sequence_encoding_transformer_hidden_size: int = 256
    sequence_encoding_transformer_heads: int = 8
    sequence_encoding_transformer_layers: int = 4
    combining_transformer_hidden_size: int = 256
    combining_transformer_heads: int = 8
    combining_transformer_layers: int = 4
    trunk_net_size: int = 576
    trunk_net_hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    add_delta_x_to_value_encoder: bool = True
    learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    dropout_rate: float = 0.1
    diffusion_loss_scale: float = 1.0
    loss_threshold: float = 100.0
    dataset_description: str = "SDE_linear_SNR_01_05_1_5_DELTA_3D"
    total_minibatch_size: int = 32
    total_minibatch_size_test: int = 64
    max_number_of_paths: int = 300
    tensorboard_figure_data: str = "test"
    plot_paths_count: int = 100
    data_loading_processes_count: int = 0

    data_in_files: DataInFiles = field(default_factory=lambda: DataInFiles(
        obs_times="obs_times.h5",
        obs_values="obs_values.h5",
        locations="hypercube_locations.h5",
        drift_at_locations="drift_functions_at_hypercube.h5",
        diffusion_at_locations="scaled_diffusion_functions_at_hypercube.h5"
    ))
    dataset_path_collections: DatasetPathCollections = field(default_factory=lambda: DatasetPathCollections(
        train=[
            "/cephfs_projects/foundation_models/data/SDE_concepts/08-09-24-with_delta_x/data-snr_01_05_1_5/linear/dim-3/1"
        ],
        test=[
            "/cephfs_projects/foundation_models/data/SDE_concepts/08-09-24-with_delta_x/data-snr_01_05_1_5/linear/dim-3/2"
        ]
    ))

def patrik_config_from_yaml(yaml_file: str) -> PatrikModelConfig:
    with open(yaml_file, 'r') as file:
        config_data = yaml.safe_load(file)

    # Initialize ModelConfig using the loaded data
    model_config = PatrikModelConfig(
        temporal_embedding_size=config_data['model'].get('temporal_embedding_size', 256),
        spatial_embedding_size=config_data['model'].get('spatial_embedding_size', 256),
        spatial_embedding_hidden_layers=config_data['model'].get('spatial_embedding_hidden_layers', None),
        sequence_encoding_transformer_hidden_size=config_data['model'].get('sequence_encoding_transformer_hidden_size', 256),
        sequence_encoding_transformer_heads=config_data['model'].get('sequence_encoding_transformer_heads', 8),
        sequence_encoding_transformer_layers=config_data['model'].get('sequence_encoding_transformer_layers', 4),
        combining_transformer_hidden_size=config_data['model'].get('combining_transformer_hidden_size', 256),
        combining_transformer_heads=config_data['model'].get('combining_transformer_heads', 8),
        combining_transformer_layers=config_data['model'].get('combining_transformer_layers', 4),
        trunk_net_size=config_data['model'].get('trunk_net_size', 576),
        trunk_net_hidden_layers=config_data['model'].get('trunk_net_hidden_layers', [256, 256]),
        add_delta_x_to_value_encoder=config_data['model'].get('add_delta_x_to_value_encoder', True),
        learning_rate=config_data['model'].get('learning_rate', 1.0e-5),
        weight_decay=config_data['model'].get('weight_decay', 1.0e-4),
        dropout_rate=config_data['model'].get('dropout_rate', 0.1),
        diffusion_loss_scale=config_data['model'].get('diffusion_loss_scale', 1.0),
        loss_threshold=config_data['model'].get('loss_threshold', 100.0),
        dataset_description=config_data['model'].get('dataset_description', "SDE_linear_SNR_01_05_1_5_DELTA_3D"),
        total_minibatch_size=config_data['model'].get('total_minibatch_size', 32),
        total_minibatch_size_test=config_data['model'].get('total_minibatch_size_test', 64),
        max_number_of_paths=config_data['model'].get('max_number_of_paths', 300),
        tensorboard_figure_data=config_data['model'].get('tensorboard_figure_data', "test"),
        plot_paths_count=config_data['model'].get('plot_paths_count', 100),
        data_loading_processes_count=config_data['model'].get('data_loading_processes_count', 0),
        data_in_files=DataInFiles(
            obs_times=config_data['model']['data_in_files']['obs_times'],
            obs_values=config_data['model']['data_in_files']['obs_values'],
            locations=config_data['model']['data_in_files']['locations'],
            drift_at_locations=config_data['model']['data_in_files']['drift_at_locations'],
            diffusion_at_locations=config_data['model']['data_in_files']['diffusion_at_locations'],
        ),
        dataset_path_collections=DatasetPathCollections(
            train=config_data['model']['dataset_path_collections']['train'],
            test=config_data['model']['dataset_path_collections']['test'],
        ),
    )
    return model_config

# Example usage
# config = load_model_config_from_yaml('path_to_your_yaml_file.yaml')

if __name__=="__main__":
    params = FIMSDEpModelParams(dim_time=30,x0_out_features=20,n_heads=1)
    print(params.encoding0_dim)
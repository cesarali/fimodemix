from lightning.pytorch.callbacks import TQDMProgressBar
import yaml
from dataclasses import asdict
from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams
from tensorboardX import SummaryWriter

def save_hyperparameters_to_yaml(hyperparams: FIMSDEpModelParams, file_path: str):
    with open(file_path, 'w') as file:
        yaml.dump(asdict(hyperparams), file)

def log_hyperparameters_to_tensorboard(hyperparams: FIMSDEpModelParams, writer: SummaryWriter):
    for key, value in asdict(hyperparams).items():
        if isinstance(value, (float, int, bool)):  # Log numeric types directly
            writer.add_text(f'hyperparams/{key}', str(value))
        elif isinstance(value, list):
            if all(isinstance(x, (int, float)) for x in value):  # Log numeric lists
                for idx, val in enumerate(value):
                    writer.add_text(f'hyperparams/{key}/{idx}', str(val))
        elif isinstance(value, str):  # Handle strings by logging as text
            writer.add_text(f'hyperparams/{key}', value)
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, List, Optional, TypeVar, Union

def check_model_devices(x):
    return x.parameters().__next__().device

def nametuple_to_device(named_tuple, device):
    return named_tuple._replace(**{
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in named_tuple._asdict().items()
    })

"""
def nametuple_to_device(obj, device):
    for attribute in vars(obj):
        value = getattr(obj, attribute)
        if isinstance(value, torch.Tensor):
            setattr(obj, attribute, value.to(device))
"""
            
def create_class_instance(class_full_path: str, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module_name, class_name = class_full_path.rsplit(".", 1)
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance

@dataclass
class GenericConfig:
    def __init__(self, data_dict: dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                setattr(self, key, GenericConfig(value))
            elif isinstance(value, tuple):
                values = []
                for v in value:
                    values.append(GenericConfig(v) if isinstance(v, dict) else v)
                setattr(self, key, values)
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        data_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, GenericConfig):
                data_dict[key] = value.to_dict()
            elif isinstance(value, list):
                values = []
                for v in value:
                    if isinstance(v, GenericConfig):
                        values.append(v.to_dict())
                    else:
                        values.append(v)
                data_dict[key] = tuple(values)
            else:
                data_dict[key] = value
        return data_dict

    def __str__(self) -> str:
        return str(self.__dict__)


def load_yaml(file_path: Path, return_object: bool = False) -> Union[dict, GenericConfig]:
    """Load a YAML file.

    Args:
        file_path (Path): existing path to the YAML file
        return_object (bool, optional): If True custom object is returned, otherwise a dictionary. Defaults to False.

    Returns:
        Union[dict, YAMLObject]: with the information in the file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.full_load(file)
    if return_object:
        return GenericConfig(config)
    return config
import logging
import pickle
from copy import copy
from datetime import datetime
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union
import yaml

def load_yaml(file_path: Path) -> dict:
    """Load a YAML file.

    Args:
        file_path (Path): existing path to the YAML file

    Returns:
        dict: with the information in the file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.full_load(file)
    return config
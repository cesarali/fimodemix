
import os
import re
import time
import json
import torch
import shutil
import subprocess
from pathlib import Path
from fimodemix import results_path
from typing import Union,Tuple,List
from dataclasses import dataclass, asdict

def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
#    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

class ExperimentsFiles:
    """
    
    """
    def __init__(self,experiment_dir=None,experiment_indentifier=None,delete=False):
        self.delete = delete
        self.define_experiment_folder(experiment_dir,experiment_indentifier)
        self.create_directories()

    def define_experiment_folder(self,experiment_dir=None,experiment_indentifier=None):
        if experiment_dir is None:
            from fimodemix import results_path
            results_dir = str(results_path)
            if experiment_indentifier is None:
                experiment_indentifier = str(int(time.time()))
            self.experiment_dir = os.path.join(results_dir, experiment_indentifier)        
        self.tensorboard_dir = os.path.join(self.experiment_dir, "logs")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
    
    def create_directories(self):
        if not Path(self.experiment_dir).exists():
            os.makedirs(self.experiment_dir)
        else:
            if self.delete:
                shutil.rmtree(self.experiment_dir)
                os.makedirs(self.experiment_dir)
            else:
                raise Exception("Folder Exist no Experiments Created Set Delete to True")
        if not os.path.isdir(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
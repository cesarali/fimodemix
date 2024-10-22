import torch
import numpy as np
from pathlib import Path
from typing import List,Optional
from dataclasses import dataclass
from collections import namedtuple
from torch.utils.data import Dataset
from fimodemix.data.generation_sde import generate_data

@dataclass
class FIMSDEpDatabatch:
    obs_values: torch.Tensor
    obs_times: torch.Tensor

    diffusion_at_hypercube: torch.Tensor
    drift_at_hypercube: torch.Tensor
    hypercube_locations: torch.Tensor

    diffusion_parameters: torch.Tensor
    drift_parameters: torch.Tensor
    process_label:torch.Tensor
    process_dimension:torch.Tensor
    
    #init_condition_distr_parameters: torch.Tensor = None
    #f_strs: torch.Tensor = None
    #g_strs: torch.Tensor = None

from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class FIMCompartementsDatabatch:

    obs_values: torch.Tensor | np.ndarray
    obs_times: torch.Tensor | np.ndarray
    obs_mask: torch.Tensor | np.ndarray
    dosing_values: torch.Tensor | np.ndarray
    dosing_times: torch.Tensor | np.ndarray
    dosing_routes: torch.Tensor | np.ndarray  # Assuming dosing routes are strings
    dosing_duration: torch.Tensor | np.ndarray
    dosing_mask: torch.Tensor | np.ndarray
    covariates: torch.Tensor | np.ndarray
    hidden_values: torch.Tensor | np.ndarray
    vector_field_at_hypercube: torch.Tensor | np.ndarray
    hypercube_locations: torch.Tensor | np.ndarray
    model_parameters: torch.Tensor | np.ndarray
    error_parameters: torch.Tensor | np.ndarray
    study_ids: torch.Tensor | np.ndarray
    hidden_process_dimension: torch.Tensor | np.ndarray
    dimension_mask: torch.Tensor | np.ndarray

    def convert_to_tensors(self):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, np.ndarray):
                try:
                    setattr(self, field, torch.tensor(value))
                except:
                    print(f"Problem for field {field}")
                    setattr(self, field, None)

# Define the named tuple
FIMSDEpDatabatchTuple = namedtuple(
    'FIMSDEpDatabatchTuple',
    [
        'obs_values',
        'obs_times',
        'diffusion_at_hypercube',
        'drift_at_hypercube',
        'hypercube_locations',
        'diffusion_parameters',
        'drift_parameters',
        'process_label',
        'process_dimension',
        'mask',
    ]
)

# Define the namedtuple which corresponds to the models databatch
FIMCompartementsDatabatchTuple = namedtuple(
    'FIMCompartementsDatabatchTuple',
    [
        'obs_values', 'obs_times','obs_mask',
        'dosing_values', 'dosing_times', 'dosing_routes', 'dosing_duration','dosing_mask',
        'covariates', 'hidden_values', 'vector_field_at_hypercube', 'hypercube_locations',
        'model_parameters', 'error_parameters',
        'study_ids', 'hidden_process_dimension','dimension_mask'
    ]
)


class FIMSDEpDataset(Dataset):
    """
    First simple dataset to train a Neural Operator 
    This Dataset performs on-the-fly dimension padding.
    """
    def __init__(self,params=None,file_paths:Optional[List[str]]=None,split="train"):
        # To keep track of the number of samples in each file
        self.data = []
        self.lengths = [] 
        # Generate and store synthetic data if no filepaths are given
        if file_paths is None:
            file_paths = self._generate_synthetic_data(split)
        # Load data and compute cumulative lengths
        self.read_files(file_paths)
        # Update Parameter Values from Dataset 
        if params is not None:
            self.update_parameters(params)

    def _generate_synthetic_data(self,split:str)->List[str]:
        """
        Generates a mix data set with lorenz and dampend oscillator
        system

        Args
            split (str) one of train, validation, test
        Returns
            filepaths (List[str])
        """
        from fimodemix import data_path
        data_path = Path(data_path)
        lorenz_path = data_path / "parameters_sde" / "lorenz_{0}.tr".format(split)
        damped_path = data_path / "parameters_sde" / "damped_{0}.tr".format(split)
        file_paths = [lorenz_path,
                      damped_path]
        if not lorenz_path.exists():
            generate_data()
        return file_paths

    def read_files(self,file_paths:List[str]):
        """
        Reads the files and organize data such that during item selection 
        the dataset points to the file and then to the location within that file
        of the particular datapoint
        """
        self.max_time_steps = 1
        self.max_dimension = 1
        self.max_hypercube_size = 1
        self.max_drift_param_size = 1
        self.max_diffusion_param_size = 1
        
        for file_path in file_paths:
            data: FIMSDEpDatabatch = torch.load(file_path)  # Adjust loading method as necessary
            self.data.append(data)
            self.lengths.append(data.obs_values.size(0))  # Number of samples in this file
            # Update max dimensions
            self.max_dimension = max(self.max_dimension, data.obs_values.size(2))
            self.max_hypercube_size = max(self.max_hypercube_size, data.diffusion_at_hypercube.size(1))
            self.max_num_steps = max(self.max_time_steps,data.obs_values.size(1))

            self.max_drift_param_size = max(self.max_drift_param_size,data.drift_parameters.size(1))
            self.max_diffusion_param_size = max(self.max_diffusion_param_size,data.diffusion_parameters.size(1))

        print(f'Max Hypercube Size: {self.max_hypercube_size}')
        print(f'Max Dimension: {self.max_dimension}')
        print(f'Max Num Steps: {self.max_num_steps}')
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)  # Total number of samples

    def __getitem__(self, idx)->FIMSDEpDatabatchTuple:
        # Obtains index of the associated file and item whithin the file
        file_idx, sample_idx = self._get_file_and_sample_index(idx)

        # Get the tensor from the appropriate file 
        obs_values = self.data[file_idx].obs_values[sample_idx]
        obs_times = self.data[file_idx].obs_times[sample_idx]
        diffusion_at_hypercube = self.data[file_idx].diffusion_at_hypercube[sample_idx]
        drift_at_hypercube = self.data[file_idx].drift_at_hypercube[sample_idx]
        hypercube_locations = self.data[file_idx].hypercube_locations[sample_idx]
        diffusion_parameters = self.data[file_idx].diffusion_parameters[sample_idx]
        drift_parameters = self.data[file_idx].drift_parameters[sample_idx]
        process_label = self.data[file_idx].process_label[sample_idx]
        process_dimension  = self.data[file_idx].process_dimension[sample_idx]
        
        # Pad and Obtain Mask of The tensors if necessary
        obs_values, diffusion_parameters, drift_parameters, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask = self._pad_tensors(
            obs_values, diffusion_parameters, drift_parameters, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations
        )
        
        # Create and return the named tuple
        return FIMSDEpDatabatchTuple(
            obs_values=obs_values,
            obs_times=obs_times,
            diffusion_at_hypercube=diffusion_at_hypercube,
            drift_at_hypercube=drift_at_hypercube,
            hypercube_locations=hypercube_locations,
            diffusion_parameters=diffusion_parameters,
            drift_parameters=drift_parameters,
            process_label=process_label,
            process_dimension=process_dimension,
            mask=mask
        )

    def _get_file_and_sample_index(self, idx):
        """Helper function to determine the file index and sample index."""
        file_idx = np.searchsorted(self.cumulative_lengths, idx,"right")
        sample_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        return file_idx, sample_idx

    def _pad_tensors(self, obs_values, diffusion_parameters, drift_parameters, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations):
        """
        Pad the tensors to ensure they meet the expected dimensions.
        it pads for
        obs_values dimension
        hypercube size, 
        drift parameter dimension,
        diffusion parameter

        Args
        
        Returns
            obs_values, diffusion_parameters, drift_parameters, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask
            mask [B,H,D] will do 0 for hypercube positions and dimensions not on batch
        """
        current_dimension = obs_values.size(1)
        current_hyper = drift_at_hypercube.size(0)

        current_diffusion = diffusion_parameters.size(0)
        current_drift = drift_parameters.size(0)

        dim_padding_size = self.max_dimension - current_dimension
        hyper_padding_size = self.max_hypercube_size - current_hyper
        drift_padding_size = self.max_drift_param_size - current_drift
        diffusion_padding_size = self.max_diffusion_param_size - current_diffusion

        if dim_padding_size > 0 or hyper_padding_size > 0 or diffusion_padding_size > 0:
            obs_values = torch.nn.functional.pad(obs_values, (0, dim_padding_size))
            diffusion_at_hypercube = torch.nn.functional.pad(diffusion_at_hypercube, (0, dim_padding_size, 0, hyper_padding_size))
            drift_at_hypercube = torch.nn.functional.pad(drift_at_hypercube, (0, dim_padding_size, 0, hyper_padding_size))
            hypercube_locations = torch.nn.functional.pad(hypercube_locations, (0, dim_padding_size, 0, hyper_padding_size))

            diffusion_parameters = torch.nn.functional.pad(diffusion_parameters, (0, diffusion_padding_size))
            drift_parameters = torch.nn.functional.pad(drift_parameters, (0, drift_padding_size))

            mask = self._create_mask(drift_at_hypercube, current_hyper, current_dimension)
        else:
            mask = torch.ones_like(obs_values)
        return obs_values, diffusion_parameters, drift_parameters, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask

    def _create_mask(self, drift_at_hypercube, current_hyper, current_dimension):
        """Create a mask for the observations.
            Args:
                drift_at_hypercube (Tensor) [B,H,D], current_hyper  (int), current_dimension (int)
            Returns:
                mask [B,H,D] will do 0 for hypercube positions and dimensions not on batch
        """
        mask = torch.ones_like(drift_at_hypercube)
        mask[current_hyper:,current_dimension:] = 0.
        return mask
    
    def update_parameters(self,param):
        param.max_dimension = self.max_dimension
        param.max_hypercube_size = self.max_hypercube_size
        param.max_num_steps = self.max_num_steps

class FIMCompartmentModels(FIMSDEpDataset):

    def __init__(self):
        super(self).__init__()
    
    def _generate_synthetic_data(self,split:str)->List[str]:
        """
        Generates a mix data set with lorenz and dampend oscillator
        system

        Args
            split (str) one of train, validation, test
        Returns
            filepaths (List[str])
        """
        from fimodemix import data_path
        data_path = Path(data_path)
        one_compartement_path = data_path / "parameters_sde" / "lorenz_{0}.tr".format(split)
        file_paths = [one_compartement_path]
        if not one_compartement_path.exists():
            generate_data()
        return file_paths

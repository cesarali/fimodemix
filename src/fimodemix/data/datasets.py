import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from fimodemix.data.generation_sde import generate_data
from dataclasses import dataclass

@dataclass
class FIMSDEDatabatch:
    obs_values: torch.Tensor
    obs_times: torch.Tensor

    diffusion_at_hypercube: torch.Tensor
    drift_at_hypercube: torch.Tensor
    hypercube_locations: torch.Tensor

    diffusion_parameters: torch.Tensor
    drift_parameters: torch.Tensor
    init_condition_distr_parameters: torch.Tensor

    f_strs: torch.Tensor = None
    g_strs: torch.Tensor = None

class FIMSDEDataset(Dataset):
    """
    This Dataset performs on-the-fly dimension padding.
    """
    def __init__(self, params=None,file_paths=None):

        if file_paths is None:
            from fimodemix import data_path
            data_path = Path(data_path)
            lorenz_path = data_path / "parameters_sde" / "lorenz.tr"
            damped_path = data_path / "parameters_sde" / "damped.tr"
            file_paths = [lorenz_path,damped_path]
            if not lorenz_path.exists():
                generate_data()
        # To keep track of the number of samples in each file
        self.data = []
        self.lengths = []  
        # Load data and compute cumulative lengths
        self.read_files(file_paths)
        if params is not None:
            self.update_parameters(params)
            
        self.cumulative_lengths = np.cumsum(self.lengths)
        
    def read_files(self, file_paths):
        self.max_time_steps = 1
        self.max_dimension = 1
        self.max_hypercube_size = 1
        for file_path in file_paths:
            data: FIMSDEDatabatch = torch.load(file_path)  # Adjust loading method as necessary
            self.data.append(data)
            self.lengths.append(data.obs_values.size(0))  # Number of samples in this file
            # Update max dimensions
            self.max_dimension = max(self.max_dimension, data.obs_values.size(2))
            self.max_hypercube_size = max(self.max_hypercube_size, data.diffusion_at_hypercube.size(1))
            self.max_num_steps = max(self.max_time_steps,data.obs_values.size(1))
        print(f'Max Hypercube Size: {self.max_hypercube_size}')
        print(f'Max Dimension: {self.max_dimension}')
        print(f'Max Num Steps: {self.max_num_steps}')

    def __len__(self):
        return sum(self.lengths)  # Total number of samples

    def __getitem__(self, idx):
        file_idx, sample_idx = self._get_file_and_sample_index(idx)

        # Get the tensor from the appropriate file 
        obs_values = self.data[file_idx].obs_values[sample_idx]
        obs_times = self.data[file_idx].obs_times[sample_idx]
        diffusion_at_hypercube = self.data[file_idx].diffusion_at_hypercube[sample_idx]
        drift_at_hypercube = self.data[file_idx].drift_at_hypercube[sample_idx]
        hypercube_locations = self.data[file_idx].hypercube_locations[sample_idx]

        # Pad the tensors if necessary
        obs_values, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask = self._pad_tensors(
            obs_values, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations
        )

        return obs_values, obs_times, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask

    def _get_file_and_sample_index(self, idx):
        """Helper function to determine the file index and sample index."""
        file_idx = np.searchsorted(self.cumulative_lengths, idx,"right")
        sample_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        return file_idx, sample_idx

    def _pad_tensors(self, obs_values, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations):
        """Pad the tensors to ensure they meet the expected dimensions."""
        current_dimension = obs_values.size(1)
        current_hyper = drift_at_hypercube.size(0)

        dim_padding_size = self.max_dimension - current_dimension
        hyper_padding_size = self.max_hypercube_size - current_hyper

        if dim_padding_size > 0 or hyper_padding_size > 0:
            obs_values = torch.nn.functional.pad(obs_values, (0, dim_padding_size))
            diffusion_at_hypercube = torch.nn.functional.pad(diffusion_at_hypercube, (0, dim_padding_size, 0, hyper_padding_size))
            drift_at_hypercube = torch.nn.functional.pad(drift_at_hypercube, (0, dim_padding_size, 0, hyper_padding_size))
            hypercube_locations = torch.nn.functional.pad(hypercube_locations, (0, dim_padding_size, 0, hyper_padding_size))
            mask = self._create_mask(obs_values, current_dimension)
        else:
            mask = torch.ones_like(obs_values)
            
        return obs_values, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask

    def _create_mask(self, obs_values, current_dimension):
        """Create a mask for the observations."""
        mask = torch.ones_like(obs_values)
        mask[:, current_dimension:] = 0.
        return mask
    
    def update_parameters(self,param):
        param.max_dimension = self.max_dimension
        param.max_hypercube_size = self.max_hypercube_size
        param.max_num_steps = self.max_num_steps

if __name__=="__main__":

    # Example usage:
    dataset = FIMSDEDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)
    databatch = next(data_loader.__iter__())
    print(len(databatch))
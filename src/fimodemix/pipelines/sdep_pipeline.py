import torch

import pandas as pd
from typing import Optional
from dataclasses import dataclass

from torch import Tensor
from fimodemix.data.datasets import FIMSDEpDatabatchTuple,FIMSDEpDatabatch
from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams
from tqdm import tqdm  # Import tqdm for the progress bar
from torch import Tensor
from typing import Tuple
from fimodemix.data.generation_sde import constant_diffusion
from fimodemix.utils.helper import (
    nametuple_to_device,
    check_model_devices
)

@dataclass
class FIMSDEPipelineOutput:

    grid:Tensor
    drift_in_grid:Tensor
    diffusion_in_grid:Tensor
    path:Tensor
    time:Tensor

class FIMSDEpPipeline:
    """
          This pipeline follows the Huggingface transformers specs 

    Inference Pipeline For SDEp 
    """
    params:FIMSDEpModelParams

    def __init__(
            self,
            model:str,
            ):
        """
        Args:
            model (FIMSDEp,string)
                if string it should download a predefined transformer model
                it takes its parameters from the model
        """
        self.model = model
        self.params = model.params

        self.num_steps = self.params.number_of_time_steps_pipeline
        self.dt = self.params.dt_pipeline
        self.device = check_model_devices(model)

    def preprocess(self, databatch):
        """sent databatch to device of model"""
        databatch = nametuple_to_device(databatch,self.device)
        return databatch

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        # outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return None
    
    def __call__(self,databatch:FIMSDEpDatabatchTuple|FIMSDEpDatabatch):
        databatch = self.preprocess(databatch) # sent to device
        paths,times = self.model_euler_maruyama_loop(databatch)
        return FIMSDEPipelineOutput(
            grid=None,
            drift_in_grid=None,
            diffusion_in_grid=None,
            path=paths,
            time=times
        )
    
    def postprocess(self, model_outputs):
        pass
    # -------------------------- SAMPLES ------------------------------------------
    def model_as_drift_n_diffusion(
        self,
        X:Tensor,
        time:Tensor=None,
        databatch:FIMSDEpDatabatchTuple=None
    )->Tuple[Tensor,Tensor]:
        """
        Defines the drift and the diffusion from the forward pass
        and handles the padding accordingly 

        Args:
            X (Tensor[B,D]): state 
            time: (None)
            databatch (FIMSDEpDatabatchTuple):
        Returns:
            drift,diffusion
        """
        D = X.size(1)
        B = X.size(0)
        X = X.unsqueeze(1) 
        # DRIFT
        drift = self.model(databatch,X).squeeze()
        # CURRENT SOLUTION DOES NOT TRAIN DIFFUSION
        diffusion = constant_diffusion(X.squeeze(),None,databatch.diffusion_parameters)
        # Create a mask based on the dimensions
        mask = torch.arange(D, device=X.device).expand(B, -1) < databatch.process_dimension  # Shape [B, D]
        # Apply the mask to X
        drift = drift * mask.float()  # Zero out elements where mask is False
        diffusion = diffusion * mask.float()  # Zero out elements where mask is False
        return drift,diffusion

    def model_euler_maruyama_step(
            self,
            states:Tensor,
            databatch:FIMSDEpDatabatchTuple
        )->Tensor:
        """
        Assumes diagonal diffusion 
        
        Args:
            states (Tensor[B,D])
            dt (float)
            model (FIMSDEp)
            databatch (databatch)
        Returns:
            new_states(Tensor[B,D])
        """
        # Calculate the deterministic part
        drift,diffusion = self.model_as_drift_n_diffusion(states,None,databatch)
        # Update the state with the deterministic part
        new_states = states + drift * self.dt
        # Add the diffusion part
        new_states += diffusion * torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(states)
        return new_states

    def model_euler_maruyama_loop(
            self,
            databatch: FIMSDEpDatabatchTuple = None,
    ):
        """
        Simulates paths from the Model using the Euler-Maruyama method.

        This is highly costly as the method needs to calculate a forward pass 
        per Euler Mayorama Step, similar cost to what one will expect in a

        Args:
            num_steps: int = 100,
            dt: float = 0.01,
            model: FIMSDEp = None,
            databatch: FIMSDEpDatabatchTuple = None,
        Returns:
            paths(Tensor[B,number_of_steps,D]),times([B,number_of_steps,D])

        """
        dimensions = databatch.obs_values.size(2)
        num_paths = databatch.obs_values.size(0)
        
        # Initialize states for all paths
        states = torch.nn.functional.sigmoid(torch.normal(0., 1., size=(num_paths, dimensions),device=self.device))

        # Store paths
        paths = torch.zeros((num_paths, self.num_steps + 1, dimensions),device=self.device)  # +1 for initial state
        paths[:, 0] = states.clone()  # Store initial states

        times = torch.linspace(0., self.num_steps * self.dt, self.num_steps + 1,device=self.device)
        times = times[None, :].repeat(num_paths, 1)

        # Simulate the paths with tqdm progress bar
        for step in tqdm(range(self.num_steps), desc="Simulating steps", unit="step"):
            states = self.model_euler_maruyama_step(states,databatch)  # Diffusion term
            paths[:, step + 1] = states.clone()  # Store new states

        return paths,times
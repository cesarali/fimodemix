import os
import time
import yaml

import torch
import shutil
import numpy as np
import torch.nn as nn
from torch import Tensor

from pathlib import Path
import pytorch_lightning as pl
from dataclasses import dataclass
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams

from fimodemix.data.datasets import (
    FIMSDEpDataset,
    FIMSDEpDatabatch,
    FIMSDEpDatabatchTuple
)

from fimodemix.data.dataloaders import (
    FIMSDEpDataLoader
)

from fimodemix.utils.experiment_files import ExperimentsFiles
from fimodemix.models.blocks import (
    TimeEncoding,
    Mlp,
    TransformerModel
)

from typing import Any, Dict, Optional, Union, List,Tuple
from dataclasses import dataclass,asdict, field
from fimodemix.trainers.utils import log_hyperparameters_to_tensorboard

# 1. Define your query generation model (a simple linear layer can work)
class QueryGenerator(nn.Module):
    def __init__(self, input_dim, query_dim):
        super(QueryGenerator, self).__init__()
        self.linear = nn.Linear(input_dim, query_dim)

    def forward(self, x):
        return self.linear(x)
    
# 2. Define a static query matrix as a learnable parameter
class StaticQuery(nn.Module):
    def __init__(self, num_steps, query_dim):
        super(StaticQuery, self).__init__()
        self.queries = nn.Parameter(torch.randn(num_steps, query_dim))  # Learnable queries

    def forward(self):
        return self.queries

# 3. Model Following FIM conventions
class FIMSDEp(pl.LightningModule):
    """
    This is the more simple architecture for 

    Stochastic Differential Equation Trainining

    """
    def __init__(
            self, 
            params: str|FIMSDEpModelParams,
            device:torch.device = None
        ):
        super(FIMSDEp, self).__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        
        self.params = params
        self._create_model(params)
        if device is not None:
            self.to(device)

        self.DatabatchNameTuple = FIMSDEpDatabatchTuple
        # Important: This property activates manual optimization (Lightning)
        self.automatic_optimization = False
        
    def _create_model(
        self,
        params: dict | FIMSDEpModelParams,
    ):
        # Architecture ---------
        self.phi_t0 = TimeEncoding(params.dim_time)

        self.phi_x0 = Mlp(in_features=params.max_dimension,
                          out_features=params.x0_out_features,
                          hidden_layers=params.x0_hidden_layers,
                          output_act=nn.SiLU())

        self.phi_1 = Mlp(in_features=params.max_dimension,
                         out_features=params.max_dimension,
                         hidden_layers=params.x0_hidden_layers)

        self.phi_2 = Mlp(in_features=params.encoding0_dim,
                         out_features=params.max_dimension,
                         hidden_layers=params.x0_hidden_layers)

        self.psi1 = TransformerModel(input_dim=params.encoding0_dim, 
                                     nhead=params.psi1_nhead, 
                                     hidden_dim=params.psi1_hidden_dim, 
                                     nlayers=params.psi1_nlayers)
        
        #self.queries = nn.Parameter(torch.randn(1, params.encoding0_dim))
        self.query_1x = QueryGenerator(input_dim=params.max_dimension,
                                       query_dim=params.encoding0_dim)

        self.query_1 =  StaticQuery(num_steps=params.max_num_steps,
                            query_dim=params.encoding0_dim)

        # Create the MultiheadAttention module
        self.omega_1 = nn.MultiheadAttention(params.encoding0_dim, params.psi1_nhead)

    def forward(
            self,
            databatch:FIMSDEpDatabatchTuple|FIMSDEpDatabatch,
            hypercube_locations:Optional[Tensor]=None,
            training:bool=True,
        ) -> Tuple[torch.tensor,torch.tensor]:
        """
        Args:

            databatch:FIMSDEpDatabatchTuple|FIMSDEpDatabatch
                keys,values:
                    hypercube_locations (Tensor [B, H, D]) 
                        where to evaluate the 
                    obs_values (Tensor [B, T, D])
                        observation values. optionally with noise. 
                    obs_times (Tensor [B, T, D])
                        observation times
                    observation_mask (dtype: bool) 
                        (0: value is observed, 1: value is masked out)
            
            hypercube_locations (tensor):
                where to evaluate the drift and diffusion function 

            training (bool): 
                flag indicating if model is in training mode. Has an impact on the output.
            
            with B: batch size, T: number of observation times, D: dimensionsm, H: number of fine grid points (locations)

        Returns:
            if training:
                f_hat (Tensor): drift estimate at hypercube points 
            else:
                dict: losses (if target drift is provided), metrics, visualizations data
        """
        obs_times = databatch.obs_times.squeeze()
        obs_values = databatch.obs_values
        if hypercube_locations is None:
            hypercube_locations = databatch.hypercube_locations.squeeze()

        batch_size = obs_times.size(0)
        num_steps = obs_times.size(1)
        dimensions = obs_values.size(2)
        num_hyper = hypercube_locations.size(1)

        # Encoding Paths -----------------
        time_encoding_ = self.phi_t0(obs_times.reshape(batch_size*num_steps,-1)) #(batch_size*num_steps,dim_time)
        x_enconding = self.phi_x0(obs_values.reshape(batch_size*num_steps,-1)) #(batch_size*num_steps,x0_out_features)
        H = torch.cat([time_encoding_,x_enconding],dim=1) #(batch_size*num_steps,encoding0_dim)
        H  = H.reshape(batch_size,num_steps,self.params.encoding0_dim) 
        H = self.psi1(torch.transpose(H,0,1)) # (seq_lenght,batch_size,encoding0_dim)

        # Trunk Queries ------------------
        hypercube_locations = hypercube_locations.reshape(batch_size*num_hyper,dimensions)
        tx = self.query_1x(hypercube_locations)  # Shape: (batch_size*num_steps, encoding0_dim)
        # Reshape queries to match the attention requirements
        tx = tx.reshape(num_hyper, batch_size, self.params.encoding0_dim)  # Shape: (num_hyper, batch_size, encoding0_dim)

        # Representation per path
        # attn_output, _ = multihead_attn(queries[:,None,:].repeat(1,batch_size,1), H, H) # Shape: (1, batch_size, query_dim)
        attn_output, _ = self.omega_1(tx, H, H) # Shape: (num_hyper, batch_size, query_dim)
        attn_output = torch.transpose(attn_output,1,0) # Shape: (num_hyper, batch_size, query_dim)
        attn_output = attn_output.reshape(num_hyper*batch_size,self.params.encoding0_dim)

        # obtain all heads
        f_hat = self.phi_2(attn_output).reshape(batch_size,num_hyper,dimensions)
        
        return f_hat

    def loss(
            self,
            f_hat:torch.tensor = None,
            g_hat:torch.tensor = None,
            f_var_hat:torch.tensor = None,
            g_var_hat:torch.tensor = None,
            mask:torch.tensor = None,
            drift_at_hypercube:torch.tensor = None,
            diffusion_at_hypercube:torch.tensor = None,
        ):
        """
        obs_values, obs_times, diffusion_at_hypercube, drift_at_hypercube, hypercube_locations, mask

        Compute the loss of the FIMODE_mix model (in original space).
            Makes sure that the mask is included in the computation of the loss

        The loss consists of supervised losses
            - negative log-likelihood of the vector field values at fine grid points
            - negative log-likelihood of the initial condition
        and an unsupervised loss
            - one-step ahead prediction loss.
        The total loss is a weighted sum of all losses. The weights are defined in the loss_configs. (loss_scale_drift, loss_scale_init_cond, loss_scale_unsuperv_loss)

        Args:
            f_hat (tuple): mean and log standard deviation of the vector field concepts (in original space) ([B, L, D], [B, L, D])
        Returns:
            dict: llh_drift, llh_init_cond, unsupervised_loss, loss = weighted sum of all losses
        """
        llh_drift = (f_hat - drift_at_hypercube)**2.
        llh_drift = llh_drift*mask
        llh_drift = llh_drift.sum(-1)
        llh_drift = llh_drift.sum(-1)
        llh_drift = torch.sqrt(llh_drift.mean())

        return llh_drift
    
    # ----------------------------- Lightning functionality ---------------------------------------------
    def prepare_batch(self,batch)->FIMSDEpDatabatchTuple:
        """lightning will convert name tuple into a full tensor for training 
        here we create the nametuple as requiered for the model
        """
        databatch = self.DatabatchNameTuple(*batch)
        return databatch
    
    def training_step(
            self, 
            batch, 
            batch_idx
    ):
        optimizer = self.optimizers()
        databatch = self.prepare_batch(batch)
        f_hats = self.forward(databatch,training=True)
        loss = self.loss(f_hats, 
                         drift_at_hypercube=databatch.drift_at_hypercube,
                         mask=databatch.mask)
        optimizer.zero_grad()
        self.manual_backward(loss)
        #if self.config.trainer.clip_grad:
        #    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.trainer.clip_max_norm)
        optimizer.step()
        #self.log('train_loss', loss)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        #self.log('discrete_training_loss', discrete_loss_.mean(), on_step=True, prog_bar=True, logger=True)
        #self.log('continuous_training_loss', continuous_loss_.mean(), on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(
        self, 
        batch, 
        batch_idx
    ):
        databatch = self.prepare_batch(batch)        
        f_hats = self.forward(databatch,training=True)
        loss = self.loss(f_hats, 
                         drift_at_hypercube=databatch.drift_at_hypercube,
                         mask=databatch.mask)
        self.log('val_loss', loss, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
    
    def on_train_start(self):
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment
        log_hyperparameters_to_tensorboard(self.params,tensorboard)

    def save_hyperparameters_to_yaml(self, file_path: str):
        with open(file_path, 'w') as file:
            yaml.dump(asdict(self.params), file)

def define_from_experiment_dir(experiment_dir):
    # define the experiment dir where everything is located
    experiment_files = ExperimentsFiles(experiment_dir,delete=False)
    checkpoint_path = experiment_files.get_lightning_checkpoint_path("best")

    # load parameters and model
    params = FIMSDEpModelParams.from_yaml(experiment_files.params_yaml)
    model = FIMSDEp.load_from_checkpoint(checkpoint_path)
    dataloaders = FIMSDEpDataLoader(params)

    return (
        model,
        dataloaders
    )
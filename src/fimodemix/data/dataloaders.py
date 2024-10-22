import logging
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import List, Optional, Union,Dict

import torch
import pandas as pd
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader,Dataset
import lightning.pytorch as pl
from fimodemix.data.datasets import (
    FIMSDEpDataset,
    FIMSDEpDatabatch,
    FIMSDEpDatabatchTuple
)
from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams


from fimodemix.data.datasets import (
    FIMCompartmentDataset,
    FIMCompartementsDatabatchTuple,
    FIMCompartementsDatabatch
)

from fimodemix.configs.config_classes.fim_compartments_config import FIMCompartmentModelParams
from fimodemix.data.generation_compartments import define_compartment_models_from_yaml

#DistributedSampler = torch.utils.data.distributed.DistributedSampler

class FIMDataloader():

    iter:Dict[str,DataLoader]
    dataset:Dict[str,Dataset]

    def __init__(self,params):
        self.params = params
        self.batch_size = params.batch_size
        self.test_batch_size = params.test_batch_size
    
    def _init_dataloaders(self, dataset):
        self.iter = {}
        for n, d in dataset.items():
            sampler = None
            #if is_distributed():
            #    sampler = DistributedSampler(d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train")
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None and n == "train",
                batch_size=batch_size,
            )
    
    @property
    def train(self):
        return self.dataset["train"]

    @property
    def train_it(self) -> DataLoader:
        return self.iter["train"]

    @property
    def validation(self):
        return self.dataset["validation"]

    @property
    def validation_it(self) -> DataLoader:
        return self.iter["validation"]

    @property
    def test(self):
        return self.dataset["test"]

    @property
    def test_it(self) -> DataLoader:
        return self.iter["test"]

    @property
    def n_train_batches(self):
        return len(self.train_it)

    @property
    def n_validation_batches(self):
        return len(self.validation_it)

    @property
    def n_test_batches(self):
        return len(self.test_it)

    @property
    def train_set_size(self):
        return len(self.train)

    @property
    def validation_set_size(self):
        return len(self.validation)

    @property
    def test_set_size(self):
        return len(self.test)

class FIMSDEpDataLoader():
    """Datalaoder for time series data in torch format."""

    def __init__(
        self,
        params:FIMSDEpModelParams
    ):
        self.params = params
        self.batch_size = params.batch_size
        self.test_batch_size = params.test_batch_size
        #self.dataset_kwargs = dataset_kwargs
        #self.loader_kwargs = loader_kwargs
        self.path = params.data_path
        self.name = params.data_name
        #self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.iter = {}
        self._init_datasets()
        self._init_dataloaders(self.dataset)

    def _init_datasets(self):
            dataset_split_names = ["train", "test", "validation"]
            self.dataset = {
                split_: FIMSDEpDataset(params=self.params,split=split_)
                for split_ in dataset_split_names
            }
    
    def _init_dataloaders(self, dataset):
        for n, d in dataset.items():
            sampler = None
            #if is_distributed():
            #    sampler = DistributedSampler(d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train")
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None and n == "train",
                batch_size=batch_size,
            )

    def __str__(self) -> str:
        #dataset_desc = {k: str(v) for k, v in self.dataset.items()}
        #dataset={dataset_desc}
        return f"TimeSeriesDataLoaderTorch=(batch_size={self.batch_size}, test_batch_size={self.test_batch_size})"

    @property
    def one_batch(self)->FIMSDEpDatabatchTuple|FIMSDEpDatabatch:
        return next(self.iter["train"].__iter__())
    
    @property
    def train(self):
        return self.dataset["train"]

    @property
    def train_it(self) -> DataLoader:
        return self.iter["train"]

    @property
    def validation(self):
        return self.dataset["validation"]

    @property
    def validation_it(self) -> DataLoader:
        return self.iter["validation"]

    @property
    def test(self):
        return self.dataset["test"]

    @property
    def test_it(self) -> DataLoader:
        return self.iter["test"]

    @property
    def n_train_batches(self):
        return len(self.train_it)

    @property
    def n_validation_batches(self):
        return len(self.validation_it)

    @property
    def n_test_batches(self):
        return len(self.test_it)

    @property
    def train_set_size(self):
        return len(self.train)

    @property
    def validation_set_size(self):
        return len(self.validation)

    @property
    def test_set_size(self):
        return len(self.test)
    
class FIMSDEpDataModule(pl.LightningDataModule):
    """wrapper for Lightning Datamodule"""
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.dataloaders = FIMSDEpDataLoader(self.params)
        
    def setup(self, stage: Optional[str] = None) -> None:
        self.dataloaders = FIMSDEpDataLoader(self.params)

    def train_dataloader(self) -> DataLoader:
        return self.dataloaders.train_it

    def val_dataloader(self) -> DataLoader:
        return self.dataloaders.validation_it

    def test_dataloader(self) -> DataLoader:
        return self.dataloaders.test_it

class FIMCompartmentDataloader(FIMDataloader):
    """
    Dataloader class for first compartment models
    """
    def __init__(self,params:FIMCompartmentModelParams):
        super().__init__(params)
        compartments_hyperparameters_file = params.compartments_hyperparameters_file
        datas = define_compartment_models_from_yaml(compartments_hyperparameters_file)
        experiment_name,train_studies,test_studies, validation_studies = datas
        self._init_datasets(
            params,train_studies,test_studies, validation_studies
        )
        self._init_dataloaders(self.dataset)

    def _init_datasets(self,params,train_studies,test_studies, validation_studies):
        train_dataset = FIMCompartmentDataset(params,
                                              train_studies)
        
        test_dataset = FIMCompartmentDataset(params,
                                              test_studies)

        validation_dataset = FIMCompartmentDataset(params,
                                                   validation_studies)
        
        self.dataset = {"train":train_dataset,
                        "test":test_dataset,
                        "validation":validation_dataset}
        
    @property
    def one_batch(self)->FIMCompartementsDatabatchTuple|FIMCompartementsDatabatch:
        return next(self.iter["train"].__iter__())
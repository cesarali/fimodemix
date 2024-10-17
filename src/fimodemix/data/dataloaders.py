import logging
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import List, Optional, Union

import torch
import pandas as pd
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader

from fimodemix.data.datasets import (
    FIMSDEpDataset,
    FIMSDEpDatabatch,
    FIMSDEpDatabatchTuple
)
from fimodemix.configs.config_classes.fim_sde_config import FIMSDEpModelParams

#DistributedSampler = torch.utils.data.distributed.DistributedSampler

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

import os
from pathlib import Path
from fimodemix.utils.helper import load_yaml

if __name__=="__main__":
    from fimodemix import config_path
    
    TRAIN_CONF = test_data_path / "config" / "ar_lstm_vanila.yaml"
    config = load_yaml(TRAIN_CONF, True)
    device_map = config.experiment.device_map
    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
import os
from functools import partial
from pathlib import Path

import click
import numpy as np
import yaml
from tqdm import tqdm

from enviroment_typing import EnvironmentConfig
from differential_equations import ExpressionGenerator
from setup_helpers import load_yaml

@click.command()
@click.option("--config", "-c", "cfg_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to the YAML configuration file", default=r"..\config\sdes_generation\ode_expressions_1d.yaml")
def main(cfg_path: Path):
    params = load_yaml(cfg_path)
    for pool_name, pool_params in params.items():
        generate_expressions(pool_name, pool_params)

def generator(x, seed, expression_conf):
    rng = np.random.RandomState([seed, os.getpid(), x])
    generator = ExpressionGenerator(expression_conf, rng)
    return generator.generate(x)

def generate_expressions(name: str, params: dict):
    n_workers = params.get("n_workers")
    chunksize = params.get("chunksize")
    num_samples = params.get("num_samples")
    seed = params.get("seed")
    output_path = Path(params.get("output_path"))

    expression_conf = EnvironmentConfig(**params.get("expression"))
    output_path_ = output_path / f"dimension_{expression_conf.min_dimension}"
    output_path_.mkdir(parents=True, exist_ok=True)

    with open(output_path_ / f"{name}.csv", "w") as f:
        pbar = tqdm(total=num_samples, desc=f"Generating {name} samples")
        for x in range(num_samples):
            result = generator(x, seed, expression_conf)
            f.write(",".join(result) + "\n")
            pbar.update()
            if pbar.n % 1000 == 0:
                f.flush()

    with open(output_path_ / f"{name}.yaml", "w") as f:
        yaml.dump(params, f)

if __name__ == "__main__":
    main()

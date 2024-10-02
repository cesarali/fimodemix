import gzip
import itertools
import json
import math
import multiprocessing
from functools import partial
from pathlib import Path

import click
from tqdm import tqdm

from setup_helpers import load_yaml


def generate_combinations(diffusion_data, drift_batch):
    return list(itertools.product(drift_batch, [diffusion_data]))


@click.command()
@click.option(
    "--config",
    "-c",
    "cfg_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML configuration file",
)

def main(cfg_path: Path):
    params = load_yaml(cfg_path)
    train_params = params.get("train")
    diffusion_pool_path = train_params["sde"]["diffusion_pool"]
    drift_pool_path = train_params["sde"]["drift_pool"]
    initial_conditions = train_params["sde"]["initial_condition"]
    hyper_grid = train_params["sde"]["hyper_grid"]
    num_samples = train_params["num_samples"]

    num_samples = int(math.sqrt(num_samples))

    with open(diffusion_pool_path, "r") as f:
        diffusion_data = [line.strip().split(",") for line in f][:num_samples]

    with open(drift_pool_path, "r") as f:
        drift_data = [line.strip().split(",") for line in f][:num_samples]

    dataset_name = params.get("dataset_name")
    batch_size = params.get("batch_size")
    n_workers = params.get("n_workers")
    chunksize = params.get("chunksize")
    output_path = Path(train_params.get("output_path"))

    total_iterations = len(drift_data) * len(diffusion_data)
    progress_bar = tqdm(total=total_iterations, desc="Generating combinations")

    output_path.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path / f"{dataset_name}.jsonl.gz", "wt", encoding="utf-8") as f:
        with multiprocessing.Pool(n_workers) as pool:
            for i in range(0, len(drift_data), batch_size):
                drift_batch = drift_data[i : i + batch_size]
                partial_generate_combinations = partial(generate_combinations, drift_batch=drift_batch)
                for j in range(0, len(diffusion_data), batch_size):
                    diffusion_batch = diffusion_data[j : j + batch_size]
                    results = pool.imap_unordered(partial_generate_combinations, diffusion_batch, chunksize=chunksize)
                    for r in itertools.chain.from_iterable(results):
                        d = {
                            "init_condition": [initial_conditions["parameters"]["mean"], initial_conditions["parameters"]["std"]],
                            "drift": r[0],
                            "diffusion": r[1],
                            "grid": hyper_grid,
                        }
                        f.write(json.dumps(d) + "\n")
                    if progress_bar.n % 1_000_000 == 0:
                        f.flush()
                    progress_bar.update(len(drift_batch) * len(diffusion_batch))

        progress_bar.close()

if __name__ == "__main__":
    main()

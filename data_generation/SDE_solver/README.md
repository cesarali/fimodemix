# SDE-Solver

## Installing Julia
```
curl -fsSL https://install.julialang.org | sh
```

## Creating training data
In order to create training data one would need to execute multiple scripts. The reason for this is that we use `drift` and `diffusion` expression generator coded in **python** (code used from [ODEFormer: Symbolic Regression of Dynamical Systems with Transformers
](https://arxiv.org/abs/2310.05573) paper). In order to solve the generated SDEs faster we use Julia.

### Workflow of generating training data

1. We generate pool of expressions for the `drift` and the `diffusion` part of the SDE. In order to do that one needs to execute the following script:

```bash
python Wiener-Procs-FM/scripts/expression_generator.py -c Wiener-Procs-FM/configs/data_generation/train/ode_expressions.yaml
```

One needs to pass a configuration file where the different pool of functions are defined. The ouput of the script is csv file for each pool of expressions where the each expression is stored as string.

**NOTE:** Not all of the `drift` expressions are solvable or fullfill specific critiria like the maximum value of the functions. Therefore, we performe a second step, pre-filtering of the expressions.

2. To filter out the solvable `drift` expressions from the generated pool of expressions, you can use the `filter_solvable_odes.jl` script. Here's how you can run it:

```bash
export JULIA_NUM_THREADS=32 # Depending on the number of threads
julia filter_solvable_odes.jl --input <input_file> --output <output_file> --dim <ode_dimension>
```

Replace `<input_file>` with the path to the CSV file containing the pool of expressions generated in the previous step. Replace `<output_file>` with the desired path and name for the filtered expressions CSV file. Replace `<ode_dimension>` with the dimension of the system.

This script will analyze each expression in the input file and filter out the ones that are not solvable or do not meet specific criteria, such as the maximum value of the functions.

Once the script finishes running, you will have a new CSV file with the filtered expressions ready for further processing.



3. Generating SDEs from Expression Pool

To generate SDEs from the pool of expressions, you can use the `sde_generator_from_pool.py` script. This script takes the filtered expressions CSV file as input and generates SDEs based on those expressions.

Here's how you can run the script:

```bash
python sde_generator_from_pool.py -c <config_file>
```

Replace `<config_file>` with the path to the config file (e.x. `configs/data_generation/train/sde_pool.yaml`).

After running the script, you will have a file containing the generated SDEs ready for further processing.

4. Finally, to solve the SDEs, run the following script:

```bash
export JULIA_NUM_THREADS=32 # Depending on the number of threads
julia wienerfm/data_generation/SDE_solver/create_training_data_pool.jl --input data/state_sde/dimension_1/state_sde.jsonl.gz --output data/ --num_paths 300 --num_samples 10000
```


## Loading the data into python
Use get_np_sde_data inside load_to_python.py

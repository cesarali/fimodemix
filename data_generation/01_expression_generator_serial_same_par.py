import re
from collections import OrderedDict
import random
import math
import click
from pathlib import Path
from typing import List

from setup_helpers import load_yaml
import importlib
expression_generator = importlib.import_module("01_expression_generator_serial")
generate_expressions = expression_generator.generate_expressions

def parameterize_expression(expression, variable=1.0):
    def replace_decimal(match):
        nonlocal param_count
        value = float(match.group())
        param = f"a_{param_count}"
        param_values[param] = value
        
        # Generate a new value from a Gaussian distribution
        new_value = random.gauss(value * variable, 1.0)
        
        param_count += 1
        return f"{new_value:.4f}"

    # Regular expression to match only decimal numbers (including scientific notation)
    decimal_pattern = r'-?\d+\.\d+(?:e[-+]?\d+)?'
    
    param_count = 0
    param_values = OrderedDict()
    parameterized_expr = re.sub(decimal_pattern, replace_decimal, expression)
    
    return parameterized_expr, param_values

def generate_multiple_expressions(base_expression: str, num_expressions: int, variables: List[float]):
    expressions = []
    for _ in range(num_expressions):
        for var in variables:
            output_expr, _ = parameterize_expression(base_expression, var)
            expressions.append(output_expr)
    return expressions

@click.command()
@click.option("--config", "-c", "cfg_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to the YAML configuration file", default=r"..\config\sdes_generation\ode_expressions_1d.yaml")
@click.option("--num-expressions", "-n", type=int, default=10, help="Number of expressions to generate for each base expression")
@click.option("--variables", "-v", type=click.FloatRange(0.1, 10.0), multiple=True, default=[0.5, 1.0, 2.0], help="Variables to use for parameterization (can be specified multiple times)")
def main(cfg_path: Path, num_expressions: int, variables: List[float]):
    params = load_yaml(cfg_path)
    
    for pool_name, pool_params in params.items():
        # Generate base expressions using the original generator
        base_expressions = generate_expressions(pool_name, pool_params)
        
        # Generate multiple parameterized expressions for each base expression
        all_expressions = []
        for base_expr in base_expressions:
            parameterized_expressions = generate_multiple_expressions(base_expr, num_expressions, variables)
            all_expressions.extend(parameterized_expressions)
        
        # Save the generated expressions
        output_path = Path(pool_params.get("output_path"))
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / f"{pool_name}_parameterized.csv", "w") as f:
            for expr in all_expressions:
                f.write(f"{expr}\n")

if __name__ == "__main__":
    main()

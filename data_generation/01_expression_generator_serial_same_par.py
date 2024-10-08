import re
from collections import OrderedDict
import random
import math

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

# Example usage
input_expr = "-1.3703 * (-0.3800 + x_0)**1 * (2)**-1 + -1.7248 * exp(-abs(0.5892 + 0.6929 * x_0))"

# Set a random seed for reproducibility
random.seed(42)

# Generate expressions with different variables
for var in [0.5, 1.0, 2.0]:
    output_expr, initial_values = parameterize_expression(input_expr, var)
    
    print(f"\nVariable: {var}")
    print("Modified expression:")
    print(output_expr)
    print("\nOriginal values:")
    for param, value in initial_values.items():
        print(f"{param} = {value}")
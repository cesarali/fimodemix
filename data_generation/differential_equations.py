from enviroment_typing import EnvironmentConfig
from odeformer.environment import FunctionEnvironment

class ExpressionGenerator:
    def __init__(self, params: EnvironmentConfig, rng: any):
        self.params = params
        self.generator = FunctionEnvironment(params)
        self.generator.rng = rng

    def generate(self, x: any, return_full: bool = False) -> list:
        expression, errors = self.generator.gen_expr(None, dimension=self.params.max_dimension, n_points=self.params.n_points)
        if return_full:
            return expression
        expression = expression["tree"]
        return [d.infix() for d in expression.nodes]

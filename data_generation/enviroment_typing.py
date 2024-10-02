from typing import Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class EnvironmentConfig(BaseModel):
    use_queue: bool = Field(default=True, description="whether to use queue")
    collate_queue_size: int = Field(default=2000)
    use_sympy: bool = Field(default=True, description="Whether to use sympy parsing (basic simplification)")
    expand: bool = Field(default=False, description="Whether to use sympy expansion")
    simplify: bool = Field(default=False, description="Whether to use further sympy simplification")
    use_abs: bool = Field(default=False, description="Whether to replace log and sqrt by log(abs) and sqrt(abs)")
    operators_to_use: str = Field(default="sin:1,inv:1,pow2:1,id:3,add:3,mul:1", description="Which operator to remove")
    operators_to_not_repeat: str = Field(default="", description="Which operator to not repeat")
    max_unary_depth: int = Field(default=7, description="Max number of operators inside unary")
    required_operators: str = Field(default="", description="Which operator to remove")
    extra_unary_operators: str = Field(default="", description="Extra unary operator to add to data generation")
    extra_binary_operators: str = Field(default="", description="Extra binary operator to add to data generation")
    extra_constants: Optional[str] = Field(default=None, description="Additional int constants floats instead of ints")
    min_dimension: int = Field(default=1)
    max_dimension: int = Field(default=2)
    max_masked_variables: int = Field(default=0)
    enforce_dim: bool = Field(default=False, description="should we enforce that we get as many examples of each dim ?")
    use_controller: bool = Field(default=True, description="should we enforce that we get as many examples of each dim ?")
    train_noise_gamma: float = Field(default=0.0, description="Should we train with additional output noise")
    eval_noise_gamma: float = Field(default=0.0, description="Should we evaluate with additional output noise")
    float_precision: int = Field(default=3, description="Number of digits in the mantissa")
    float_descriptor_length: int = Field(default=3, description="Type of encoding for floats")
    max_exponent: int = Field(default=100, description="Maximal order of magnitude")
    max_trajectory_value: float = Field(default=1e2, description="Maximal value in trajectory")
    discard_stationary_trajectory_prob: float = Field(default=0.9, description="Probability to discard stationary trajectories")
    max_prefactor: int = Field(default=20, description="Maximal order of magnitude in prefactors")
    max_token_len: int = Field(default=0, description="max size of tokenized sentences, 0 is no filtering")
    tokens_per_batch: int = Field(default=10000, description="max number of tokens per batch")
    pad_to_max_dim: bool = Field(default=True, description="should we pad inputs to the maximum dimension?")
    use_two_hot: bool = Field(default=False, description="Whether to use two hot embeddings")
    max_int: int = Field(default=10, description="Maximal integer in symbolic expressions")
    min_binary_ops_per_dim: int = Field(default=1, description="Min number of binary operators per input dimension")
    max_binary_ops_per_dim: int = Field(default=5, description="Max number of binary operators per input dimension")
    min_unary_ops_per_dim: int = Field(default=0, description="Min number of unary operators")
    max_unary_ops_per_dim: int = Field(default=3, description="Max number of unary operators")
    min_op_prob: float = Field(
        default=0.01, description="Minimum probability of generating an example with given n_op, for our curriculum strategy"
    )
    max_points: int = Field(default=200, description="Max number of terms in the series")
    min_points: int = Field(default=50, description="Min number of terms per dim")
    n_points: int = Field(default=100, description="Number of points in the series")
    prob_const: float = Field(default=0.0, description="Probability to generate const in leafs")
    prob_prefactor: float = Field(default=1, description="Probability to generate prefactor")
    reduce_num_constants: bool = Field(default=True, description="Use minimal amount of constants in eqs")
    use_skeleton: bool = Field(default=False, description="should we use a skeleton rather than functions with constants")
    prob_rand: float = Field(default=0.0, description="Probability to generate n in leafs")
    time_range: float = Field(default=10.0, description="Time range for ODE integration")
    prob_t: float = Field(default=0.0, description="Probability to generate n in leafs")
    train_subsample_ratio: float = Field(default=0.5, description="Ratio of timesteps to remove")
    eval_subsample_ratio: float = Field(default=0, description="Ratio of timesteps to remove")
    ode_integrator: str = Field(default="solve_ivp", description="ODE integrator to use")
    solve_ode: bool = Field(default=False, description="Whether to solve the ODE")
    init_scale: float = Field(default=1.0, description="Scale for initial conditions")
    fixed_init_scale: bool = Field(default=False, description="Fix the init scale")
    n_words: Optional[int] = Field(default=None, description="Number of words in the dictionary")
    c_min: Optional[float] = Field(default=0.001, description="Min value for prefactor")
    c_max: Optional[float] = Field(default=1, description="Max value for prefactor")
    debug: bool = Field(default=False, description="Debug mode")
    skip_linear: bool = Field(default=False, description="Skip linear equations")
    sample_type: str = Field(
        default="exponent",
        description="Sample distribution type for the linear transformation. Possible values are 'exponent', 'uniform' and 'normal'",
    )

class FunctionExpressionParameters(BaseModel):
    operators: str
    max_ops: int
    max_int: int
    max_len: int
    int_base: int
    balanced: bool
    precision: int
    positive: bool
    rewrite_functions: str
    leaf_probs: str
    n_variables: int
    n_coefficients: int

    @field_validator("n_variables")
    def val_n_variables(cls, v: int, info: ValidationInfo):
        if v < 1 and v > 3:
            raise ValueError("n_variables must be between 1 and 3")
        return v

    @field_validator("n_coefficients")
    def val_n_coefficients(cls, v: int, info: ValidationInfo):
        if v < 0 and v > 10:
            raise ValueError("n_coefficients must be between 0 and 10")
        return v


from typing import List
from dataclasses import dataclass, field

@dataclass
class FIMSDEModelParams:
    # data
    input_size: int = 1  # Original input size

    max_dimension:int = 1
    max_hypercube_size:int = 1
    max_num_steps:int = 1

    # model architecture
    dim_time:int = 19

    # phi_0 / first data encoding
    x0_hidden_layers: List[int] = field(default_factory=lambda:[50,50])
    x0_out_features: int = 21
    x0_dropout: float = 0.2

    encoding0_dim:int = 40 #  x0_out_features + dim_time

    #psi_1 / first transformer
    psi1_nhead:int = 2
    psi1_hidden_dim:int = 300
    psi1_nlayers:int = 2

    #Multiheaded Attention 1 / first path summary
    query_dim:int = 10

    n_heads: int = 4
    hidden_dim: int = 64
    output_size: int = 1
    batch_size: int = 32
    seq_length: int = 10

    # training
    num_epochs: int = 10
    learning_rate: float = 0.001
    embed_dim: int = 8  # New embedding dimension

    def __post__init__(self):
        self.encoding0_dim = self.x0_out_features + self.dim_time
import torch
import torch.nn as nn
import torch.distributed as dist

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
def is_distributed() -> bool:
    return dist.is_initialized()
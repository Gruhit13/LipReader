import torch as T
from torch import nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int):
        super(GLU, self).__init__()
        self.dim = dim
    
    def forward(self, input: T.Tensor) -> T.Tensor:
        output, gate = input.chunk(2, dim=self.dim)
        return output * gate.sigmoid() 
import torch as T
from torch import nn
from typing import Tuple

class Transpose(nn.Module):
    def __init__(self, shape: Tuple):
        super(Transpose, self).__init__()
        self.shape = shape
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        oup = x.transpose(*self.shape)
        return oup
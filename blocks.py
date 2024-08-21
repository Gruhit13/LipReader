import torch as T
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class ResnetBlock(nn.Module):
    def __init__(self, in_feature: int, out_feature: int, kernel_size: int, stride: int = 1, padding: int = 1):
        super(ResnetBlock, self).__init__()

        # Main Layer
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_feature, out_channels=out_feature, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_feature)
        )

        # downsampler
        self.identify = True

        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_feature)
            )
            self.identify = False
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        
        residual = x
        oup = self.layers(x)

        if not self.identify:
            residual = self.downsample(x)
        
        oup = oup + residual
        oup = F.relu(oup)

        return oup

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
        
    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.linear(x)

class GlobalAvgPool2d(nn.Module):
    def __init__(self, dim=Tuple[int], keepdim: bool = False):
        super(GlobalAvgPool2d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        x = x.mean(dim=self.dim, keepdim=self.keepdim)
        return x

class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "Out channels must be divisible by in_channels"

        self.depthwise_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups = in_channels,
            stride = stride,
            padding = padding,
            bias = bias
        )
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.depthwise_conv(x)

class PointwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super(PointwiseConv1d, self).__init__()
        self.pointwise_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride = stride,
            padding = padding,
            bias = bias
        )
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.pointwise_conv(x)
    

class ResidualConnectionBlock(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        module_factor: float,
        input_factor: float = 1.0
    ):
        super(ResidualConnectionBlock, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor
    
    def forward(self, *x:T.Tensor) -> T.Tensor:
        return (self.module(*x) * self.module_factor) + (x[0] * self.input_factor)
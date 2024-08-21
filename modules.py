import torch as T
from torch import nn
from torch.nn import functional as F
from blocks import ResnetBlock, PointwiseConv1d, DepthwiseConv1d, ResidualConnectionBlock, Linear, GlobalAvgPool2d
from attention import MultiHeadSelfAttention
from typing import List, Optional
from transform import Transpose
from activations import Swish, GLU

###################### Resnet Front End ######################
# This is an extremly simple Resnet architecture based       #
# frontend that will be used to extract feature framewise.   #
##############################################################

class ResNetFrontEnd(nn.Module):
    def __init__(self, layers: List[int]):
        super(ResNetFrontEnd, self).__init__()
        
        # Expecting that the input is of shape B x 64 X H X W
        self.block1 = self.__make_block(layers[0], 64, 64)
        self.block2 = self.__make_block(layers[1], 64, 128)
        self.block3 = self.__make_block(layers[2], 128, 256)
        self.avgpool = GlobalAvgPool2d(dim=(2, 3))
        self.linear = Linear(in_features=256, out_features=256)

    
    def __make_block(self, layers: int, in_feature: int, out_features: int) -> nn.Module:
        # 1st section of every block will have a downsampling block
        first_block = ResnetBlock(in_feature, out_features, kernel_size=3, stride=2, padding=1)

        layers_list = [first_block]
        for i in range(1, layers):
            layers_list.append(ResnetBlock(out_features, out_features, kernel_size=3, stride=1, padding=1))
        
        block = nn.Sequential(*layers_list)
        return block

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = self.linear(x)

        return x

###################### FeedForward Network ######################
# This is a feed forward network with pre-normalization as     #
# shown in the paper. Also it uses swish activation function   #
################################################################

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        expansion_factor: int,
        dropout: float = 0.1
    ):
        super(FeedForwardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim*expansion_factor, bias=True),
            Swish(),
            nn.Dropout(dropout),
            Linear(encoder_dim*expansion_factor, encoder_dim, bias=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, x:T.Tensor) -> T.Tensor:
        return self.network(x)

###################### Convolution Block ######################
# This is the entire convolution network that performs all    #
# convolution related operations. First pointwise convolution #
# to expand dimension then GLU activation then depthwise      #
# convolution and lastly again a Pointwise convolution        #
###############################################################

class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_expand: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        stride: int = 1,
        dropout: float = 0.1
    ):
        super(ConvolutionBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be a odd number for padding to be 'SAME'"

        self.conv_block = nn.Sequential(
            nn.LayerNorm(dim_model),
            Transpose((1, 2)), # This convert [B, T, D] => [B, D, T]
            PointwiseConv1d(dim_model, expansion_factor*dim_expand, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(dim_expand, dim_expand, kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(dim_expand),
            Swish(),
            PointwiseConv1d(dim_expand, dim_expand, stride=1, padding=0, bias=True),
            nn.Dropout(dropout),
            Transpose((1, 2)), # This convert [B, D, T] => [B, T, D]
        )
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.conv_block(x)

###################### Conformer Block ######################
# A full conformer block that merges all the layers mention #
# in paper. Thus creating a full single conformer block     #
# functionality.                                            #
#############################################################

class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_expand: int,
        max_seq_len: int,
        ff_expansion_factor: int = 4,
        ff_residual_factor: float = 0.5,
        num_heads: int = 4,
        kernel_size: int = 31,
        stride: int = 1,
        conv_expansion_factor: int = 2,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ):
        super(ConformerBlock, self).__init__()

        self.ff_residual_factor = ff_residual_factor
        
        # First Half residual FeedForward Block
        self.ff_net_1 = ResidualConnectionBlock(
            FeedForwardNetwork(
                dim_model,
                ff_expansion_factor,
                ff_dropout
            ),
            module_factor = self.ff_residual_factor
        )

        # MHSA residual block
        self.mhsa = ResidualConnectionBlock(
            MultiHeadSelfAttention(
                max_seq_len,
                dim_model,
                num_heads,
                attn_dropout
            ),
            module_factor = 1.
        )

        # Convolution residual block
        self.conv = ConvolutionBlock(
            dim_model,
            dim_expand,
            kernel_size,
            conv_expansion_factor,
            stride,
            conv_dropout
        )

        # This is when the temporal dimension is downsample at that time to handle the 
        # residual connection. Similar to the ResNet architecture
        self.conv_residual = False
        if dim_model != dim_expand:
            self.conv_residual = True
            self.conv_res = nn.Sequential(
                Transpose((1, 2)),
                nn.Conv1d(dim_model, dim_expand, kernel_size=1, stride=stride),
                Transpose((1, 2))
            )

        # Second Half residual FeedForward Block
        self.ff_net_2 = ResidualConnectionBlock(
            FeedForwardNetwork(
                dim_expand,
                ff_expansion_factor,
                ff_dropout
            ),
            module_factor = self.ff_residual_factor
        )
    
    def forward(self, x: T.Tensor, mask: Optional[T.Tensor]=None) -> T.Tensor:
        x = self.ff_net_1(x)
        x = self.mhsa(x, mask)
        
        if self.conv_residual:
            x = self.conv_res(x) + self.conv(x)
        else:
            x = x + self.conv(x)
        
        x = self.ff_net_2(x)
        return x
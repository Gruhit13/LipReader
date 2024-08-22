from typing import Tuple, List

class VECConfig:
    vocab_size: int = 256
    video_processing_in_channels: int = 1
    video_processing_out_channels: int = 64
    video_processing_kernel_size: Tuple[int] = (5, 7, 7)
    video_processing_stride: Tuple[int] = (1, 2, 2)
    resnet_layers: List[int] = [2, 2, 2]
    dim_model: List[int] = [256, 360]
    num_blocks: List[int] = [6, 1]
    interctc_block: List[int] = [3, 6]
    max_seq_len: int = 10000
    ff_expansion_factor: int = 4
    ff_residual_factor: float = 0.5
    num_heads: int = 4
    kernel_size: int = 15
    conv_expansion_factor: int = 2
    ff_dropout: float = 0.1
    attn_dropout: float = 0.1
    conv_dropout: float = 0.1
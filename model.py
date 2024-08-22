import torch as T
from torch import nn
from config import VECConfig
import transform
from modules import ResNetFrontEnd
from networks import Conformer
from blocks import Linear
from typing import Optional

class VisualEfficientConformer(nn.Module):
    def __init__(
        self,
        config: VECConfig
    ):
        super(VisualEfficientConformer, self).__init__()

        self.pre_padding = nn.ConstantPad3d(
            padding=(
                (config.video_processing_kernel_size[2]-1) // 2, # Left
                config.video_processing_kernel_size[2] // 2, # right

                (config.video_processing_kernel_size[1] - 1) // 2, # top
                config.video_processing_kernel_size[1] // 2, # down

                (config.video_processing_kernel_size[0] - 1) // 2, # front
                config.video_processing_kernel_size[0] // 2, # back
            ),
            value = 0 
        )

        self.video_preprocessor = nn.Sequential(
            nn.Conv3d(
                in_channels = config.video_processing_in_channels,
                out_channels = config.video_processing_out_channels,
                kernel_size = config.video_processing_kernel_size,
                stride = config.video_processing_stride,
            ),
            nn.BatchNorm3d(config.video_processing_out_channels),
            nn.ReLU()
        )

        self.video_to_image = transform.VideoToImage()

        
        self.resnet = ResNetFrontEnd(config.resnet_layers)

        self.image_to_video = transform.ImageToVideo()
        
        # Conformer model to process the sequence data. 
        self.conformer = Conformer(
            vocab_size = config.vocab_size,
            dim_model = config.dim_model,
            num_blocks = config.num_blocks,
            interctc_block = config.interctc_block,
            max_seq_len = config.max_seq_len,
            ff_expansion_factor = config.ff_expansion_factor,
            ff_residual_factor = config.ff_residual_factor,
            num_heads = config.num_heads,
            kernel_size = config.kernel_size,
            conv_expansion_factor = config.conv_expansion_factor,
            ff_dropout = config.ff_dropout,
            attn_dropout = config.attn_dropout,
            conv_dropout = config.conv_dropout
        )

        # Attach the final classification head
        self.head = Linear(in_features=config.dim_model[-1], out_features=config.vocab_size)
    
    def forward(self, input: T.Tensor, lengths: Optional[T.Tensor]=None, mask: Optional[T.Tensor] = None) -> T.Tensor:
        # Input video frames will be of shape [B, C, T, H, W]

        video_frames = input.size(2)

        input = self.pre_padding(input)

        # Firstvideo_frames the video processing layer
        # [B, C, T, H, W] => [B, 64, T', H', W']
        x = self.video_preprocessor(input)

        # Convert Video to image [B, 64, T', H', W'] => [BT', 64, H', W']
        x = self.video_to_image(x)

        # Process Images with ResNet Frontend model
        # [BT', 64, H', W'] => [BT', 256]
        x = self.resnet(x)

        # [BT', 256] => [BT', 256, 1, 1]
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # [BT', 256, 1, 1] => [B, 256, T', 1, 1]
        x = self.image_to_video(x, video_frames)

        # [B, 256, T', 1, 1] => [B, 256, T'] => [B, T', 256]
        x = x.squeeze(-1).squeeze(-1).transpose(1, 2)

        # [B, T', 256] => [B, T'', 360]
        x, interctc_output = self.conformer(x, mask)

        # [B, T'', 360] => [B, T'', 360]
        x = self.head(x)

        return x, interctc_output

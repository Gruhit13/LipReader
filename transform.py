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

class VideoToImage(nn.Module):
    def __init__(self):
        super(VideoToImage, self).__init__()
    
    def forward(self, video: T.Tensor) -> T.Tensor:
        # [B, C, T, H, W] => [B, T, C, H, W] => [BT, C, H, W]
        return video.transpose(1, 2).flatten(start_dim=0, end_dim=1)

class ImageToVideo(nn.Module):
    def __init__(self):
        super(ImageToVideo, self).__init__()
    
    def forward(self, images: T.Tensor, video_frames: int) -> T.Tensor:
        assert images.size(0) % video_frames == 0, "Number of elements at dim-0 must be divisible by video_frame"

        # [BT, C, H, W] => [B, T, C, H, W] => [B, C, T, H, W]
        return images.view(images.size(0) // video_frames, video_frames, images.size(1), images.size(2), images.size(3)).transpose(1, 2)

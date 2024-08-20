import torch as T
from torch import nn

class RelativeSinusoidalPositionEncoding(nn.Module):
    def __init__(self, max_seq_len: int, dim_model: int):
        super(RelativeSinusoidalPositionEncoding, self).__init__()

        pos_encoding = T.zeros(2*max_seq_len - 1, dim_model)

        rel_pos_left = T.arange(start=max_seq_len-1, end=0, step=-1, dtype=T.float)
        rel_pos_right = T.arange(start=0, end=-max_seq_len, step=-1, dtype=T.float)
        pos = T.concat([rel_pos_left, rel_pos_right], dim=0).unsqueeze(1)

        angles = pos / 10000**(2 * T.arange(0, dim_model//2, dtype=T.float).unsqueeze(0) / dim_model)

        # Set the relative position encoding
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)
        self.max_seq_len = max_seq_len
        self.dim_model = dim_model
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        bt_size, seq_len, _ = x.shape

        E = self.pos_encoding[:, self.max_seq_len - seq_len: self.max_seq_len-1 + seq_len]

        E = E.repeat(bt_size, 1, 1)
        return E

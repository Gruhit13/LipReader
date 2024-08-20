import torch as T
from torch import nn
from torch.nn import functional as F
from typing import Optional
import math
from embeddings import RelativeSinusoidalPositionEncoding
from blocks import Linear

class RelativeMultiheadAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super(RelativeMultiheadAttention, self).__init__()

        assert dim_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = self.dim_model // self.num_heads
        self.sqrt_dim = math.sqrt(self.head_dim)

        self.query_layer = Linear(self.dim_model, self.dim_model, bias=False)
        self.key_layer = Linear(self.dim_model, self.dim_model, bias=False)
        self.value_layer = Linear(self.dim_model, self.dim_model, bias=False)
        self.pos_proj = Linear(self.dim_model, self.dim_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.u_bias = nn.Parameter(T.Tensor(self.num_heads, self.head_dim))
        self.v_bias = nn.Parameter(T.Tensor(self.num_heads, self.head_dim))
        T.nn.init.xavier_uniform_(self.u_bias)
        T.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(self.dim_model, self.dim_model, bias=False)
    
    def forward(
        self, 
        query: T.Tensor, 
        key: T.Tensor, 
        value: T.Tensor, 
        pos_encoding: T.Tensor,
        mask: Optional[T.Tensor] = None
    ) -> T.Tensor:
        batch_size = query.size(0)

        # [B, T, D] => [B, T, nH, HD]
        query = self.query_layer(query).view(batch_size, -1, self.num_heads, self.head_dim)

        # [B, T, D] => [B, T, nH, HD] => (after pemute) => [B, nH, T, HD]
        key = self.key_layer(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value_layer(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # [B, 2*T-1, D] => [B, 2*T-1, nH, HD]
        pos_encoding = self.pos_proj(pos_encoding).view(batch_size, -1, self.num_heads, self.head_dim)

        # [B, nH, T, HD] x [B, nH, HD, T] => [B, nH, T, T]
        content_score = T.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))

        # [B, nH, T, HD] x [B, nh, HD, 2*T - 1] => [B, nH, T, 2*T - 1]
        pos_score = T.matmul((query + self.v_bias).transpose(1, 2), pos_encoding.permute(0, 2, 3, 1))
        
        # [B, nH, T, 2*T - 1] => [B, nH, T, T]
        pos_score = self.__relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            # Expecting mask to be a 3-D tensor of shape [B, T, T]
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask, -1e9)

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # [B, nH, T, T] x [B, nH, T, D] => [B, nH, T, D] => [B, T, nH, D]
        context = T.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.dim_model)
        return self.out_proj(context)

    def __relative_shift(self, pos_score: T.Tensor) -> T.Tensor:
        # pos_score shape is [B, nH, T, 2*T-1]

        bt_size, n_heads, seq_length1, seq_length2 = pos_score.size()

        # creates tensor of shape [B, nH, T, 1]
        zeros = pos_score.new_zeros((bt_size, n_heads, seq_length1, 1))

        # [B, nH, T, 2*T-1] => [B, nH, T, 2*T]
        padded_pos_score = T.cat([zeros, pos_score], dim=-1)

        # [B, nH, T, 2*T] => [B, nH, 2*T, T]
        padded_pos_score = padded_pos_score.view(bt_size, n_heads, seq_length2+1, seq_length1)

        # [B, nH, 2*T, T] => (after shifting left) [B, nH, 2*T-1, T] => (after view as) [B, nH, T, 2*T-1] => (after last slicing) [B, nH, T, T]
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, max_seq_len: int, dim_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(dim_model)
        self.rel_pos_embedding = RelativeSinusoidalPositionEncoding(max_seq_len, dim_model)
        self.attention = RelativeMultiheadAttention(dim_model, num_heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:T.Tensor, mask: Optional[T.Tensor]) -> T.Tensor:
        x = self.layer_norm(x)

        E = self.rel_pos_embedding(x)
        output = self.attention(x, x, x, pos_encoding=E, mask=mask)
        output = self.dropout(output)
        return output

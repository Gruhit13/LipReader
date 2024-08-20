import torch as T
from torch import nn
from typing import List, Optional
from modules import ConformerBlock

class Conformer(nn.Module):
    def __init__(
        self,
        dim_model: List[int],
        num_blocks: List[int],
        max_seq_len: int,
        ff_expansion_factor: int = 4,
        ff_residual_factor: float = 0.5,
        num_heads: int = 4,
        kernel_size: int = 31,
        conv_expansion_factor: int = 2,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_dropout: float = 0.1
    ):
        super(Conformer, self).__init__()

        assert len(dim_model) == len(num_blocks), "Length of dimension must match length of number of blocks"

        self.blocks = nn.ModuleList()
        # Process stagewise
        for stage_id in range(len(num_blocks)):

            # Process each block in stage
            for block_id in range(num_blocks[stage_id]):
                
                down_block = (block_id == num_blocks[stage_id]-1) and (stage_id < len(num_blocks)-1)

                self.blocks.append(
                    ConformerBlock(
                        dim_model = dim_model[stage_id],
                        dim_expand = dim_model[stage_id + (1 if down_block else 0)],
                        max_seq_len = max_seq_len,
                        ff_expansion_factor = ff_expansion_factor,
                        ff_residual_factor = ff_residual_factor,
                        num_heads = num_heads,
                        kernel_size = kernel_size,
                        stride = 1 if not down_block else 2,
                        conv_expansion_factor = conv_expansion_factor,
                        ff_dropout = ff_dropout,
                        attn_dropout = attn_dropout,
                        conv_dropout = conv_dropout
                    )
                )
            
            
    
    def forward(self, x: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        
        return x
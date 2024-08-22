import torch as T
from torch import nn
from typing import List, Optional
from modules import ConformerBlock
from blocks import InterCTCBlock

class Conformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_model: List[int],
        num_blocks: List[int],
        interctc_block: List[int],
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

        self.interctc_block = interctc_block
        self.blocks = nn.ModuleList()
        self.interctc_module = nn.ModuleList()
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

                if block_id+1 in interctc_block:
                    self.interctc_module.append(
                        InterCTCBlock(dim_model=dim_model[stage_id + (1 if down_block else 0)], vocab_size=vocab_size)
                    )

    
    def forward(self, x: T.Tensor, length: Optional[T.Tensor] = None, mask: Optional[T.Tensor] = None) -> T.Tensor:
        interctc_output = {}
        interctc_block_index = 0
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask)

            if i+1 in self.interctc_block:
                
                x, logits = self.interctc_module[interctc_block_index](x)
                interctc_block_index += 1 # Increase the index to fetch from next interctc block
                key = f"inter_ctc_at_{i+1}"

                if block.stride > 1 and length is not None:
                    length = T.div(length - 1, block.stride, rounding_mode='floor') + 1
                
                interctc_output[key] = [logits, length]
        
        return x, interctc_output
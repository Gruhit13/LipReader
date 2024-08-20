import torch as T
from networks import Conformer
from torchsummary import summary

layers = [2, 2, 2]
batch_size = 2
frame_height = 64
frame_width = 64
channel_dim = 3
timestamps = 32
dim_model = [64, 128]
num_blocks = [2, 2]
num_heads = 4

conformer = Conformer(
    dim_model= dim_model,
    num_blocks=num_blocks,
    max_seq_len=timestamps,
    ff_expansion_factor=4,
    ff_residual_factor=0.5,
    num_heads=num_heads,
    kernel_size=15,
    conv_expansion_factor=2,
    ff_dropout=0.1,
    attn_dropout=0.1,
    conv_dropout=0.1
)

inp_seq = T.randn(batch_size, timestamps, dim_model[0])
print("Input Shape: ", inp_seq.shape)
oup = conformer(inp_seq)
print("Output Shape:", oup.shape)

model_summary = summary(conformer, input_data=inp_seq, depth=4, verbose=0)
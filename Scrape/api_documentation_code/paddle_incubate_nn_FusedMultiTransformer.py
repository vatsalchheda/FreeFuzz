# required: gpu
import paddle
from paddle.incubate.nn import FusedMultiTransformer

# encoder input: [batch_size, src_len, d_model]
enc_input = paddle.rand((2, 4, 128))
# self attention mask: [batch_size, 1, src_len, src_len]
attn_mask = paddle.rand((2, 1, 4, 4))
encoder_layers = FusedMultiTransformer(128, 2, 512, num_layers=1)
enc_output = encoder_layers(enc_input, attn_mask)  # [2, 4, 128]
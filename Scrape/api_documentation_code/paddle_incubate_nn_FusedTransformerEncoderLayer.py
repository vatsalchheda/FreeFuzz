# required: gpu
import paddle
from paddle.incubate.nn import FusedTransformerEncoderLayer

# encoder input: [batch_size, src_len, d_model]
enc_input = paddle.rand((2, 4, 128))
# self attention mask: [batch_size, n_head, src_len, src_len]
attn_mask = paddle.rand((2, 2, 4, 4))
encoder_layer = FusedTransformerEncoderLayer(128, 2, 512)
enc_output = encoder_layer(enc_input, attn_mask)  # [2, 4, 128]
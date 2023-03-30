# required: gpu
import paddle
# input: [batch_size, sequence_length, embed_dim]
query = paddle.rand((2, 4, 128))
# self attention mask: [batch_size, num_heads, query_len, query_len]
attn_mask = paddle.rand((2, 2, 4, 4))
multi_head_attn = paddle.incubate.nn.FusedMultiHeadAttention(128, 2)
output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
import paddle
arg_1_tensor = paddle.rand([3, 5, 9, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 2, 64, 128], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([128, 128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = True
arg_5 = None
arg_6 = None
arg_7_tensor = paddle.rand([], dtype=paddle.float32)
arg_7 = arg_7_tensor.clone()
arg_8_tensor = paddle.rand([128], dtype=paddle.float32)
arg_8 = arg_8_tensor.clone()
arg_9 = True
arg_10_tensor = paddle.rand([3, 2, 64], dtype=paddle.float32)
arg_10 = arg_10_tensor.clone()
arg_11_tensor = paddle.rand([128], dtype=paddle.float32)
arg_11 = arg_11_tensor.clone()
arg_12 = None
arg_13_tensor = paddle.rand([2, 2, 4, 0], dtype=paddle.float32)
arg_13 = arg_13_tensor.clone()
arg_14 = 0.5
arg_15 = -1.0
arg_16 = 1.00001
arg_17 = True
arg_18 = -1
arg_19 = None
res = paddle.incubate.nn.functional.fused_multi_head_attention(x=arg_1,qkv_weight=arg_2,linear_weight=arg_3,pre_layer_norm=arg_4,pre_ln_scale=arg_5,pre_ln_bias=arg_6,ln_scale=arg_7,ln_bias=arg_8,pre_ln_epsilon=arg_9,qkv_bias=arg_10,linear_bias=arg_11,cache_kv=arg_12,attn_mask=arg_13,dropout_rate=arg_14,attn_dropout_rate=arg_15,ln_epsilon=arg_16,training=arg_17,ring_id=arg_18,name=arg_19,)

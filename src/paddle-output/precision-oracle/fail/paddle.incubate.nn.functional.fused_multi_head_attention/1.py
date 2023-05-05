results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 4, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([3, 0, 0, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([128, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = False
arg_5 = None
arg_6 = None
float_tensor = paddle.rand([128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_7_tensor = f16_tensor
arg_7 = arg_7_tensor.clone()
float_tensor = paddle.rand([128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_8_tensor = f16_tensor
arg_8 = arg_8_tensor.clone()
arg_9 = 1e-05
float_tensor = paddle.rand([3, 2, 64], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_10_tensor = f16_tensor
arg_10 = arg_10_tensor.clone()
float_tensor = paddle.rand([128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_11_tensor = f16_tensor
arg_11 = arg_11_tensor.clone()
arg_12 = None
float_tensor = paddle.rand([2, 2, 4, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_13_tensor = f16_tensor
arg_13 = arg_13_tensor.clone()
arg_14 = 0.1
arg_15 = 0.1
arg_16 = 1e-05
arg_17 = True
arg_18 = -1
arg_19 = None
start = time.time()
results["time_low"] = paddle.incubate.nn.functional.fused_multi_head_attention(x=arg_1,qkv_weight=arg_2,linear_weight=arg_3,pre_layer_norm=arg_4,pre_ln_scale=arg_5,pre_ln_bias=arg_6,ln_scale=arg_7,ln_bias=arg_8,pre_ln_epsilon=arg_9,qkv_bias=arg_10,linear_bias=arg_11,cache_kv=arg_12,attn_mask=arg_13,dropout_rate=arg_14,attn_dropout_rate=arg_15,ln_epsilon=arg_16,training=arg_17,ring_id=arg_18,name=arg_19,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_7 = arg_7_tensor.clone().type(paddle.float32)
arg_8 = arg_8_tensor.clone().type(paddle.float32)
arg_10 = arg_10_tensor.clone().type(paddle.float32)
arg_11 = arg_11_tensor.clone().type(paddle.float32)
arg_13 = arg_13_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.incubate.nn.functional.fused_multi_head_attention(x=arg_1,qkv_weight=arg_2,linear_weight=arg_3,pre_layer_norm=arg_4,pre_ln_scale=arg_5,pre_ln_bias=arg_6,ln_scale=arg_7,ln_bias=arg_8,pre_ln_epsilon=arg_9,qkv_bias=arg_10,linear_bias=arg_11,cache_kv=arg_12,attn_mask=arg_13,dropout_rate=arg_14,attn_dropout_rate=arg_15,ln_epsilon=arg_16,training=arg_17,ring_id=arg_18,name=arg_19,)
results["time_high"] = time.time() - start

print(results)

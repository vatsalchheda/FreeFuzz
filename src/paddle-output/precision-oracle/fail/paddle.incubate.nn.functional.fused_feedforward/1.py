results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 4, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([128, 512], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([512, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
float_tensor = paddle.rand([512], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_tensor = f16_tensor
arg_4 = arg_4_tensor.clone()
float_tensor = paddle.rand([128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_tensor = f16_tensor
arg_5 = arg_5_tensor.clone()
arg_6 = None
arg_7 = None
float_tensor = paddle.rand([128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_8_tensor = f16_tensor
arg_8 = arg_8_tensor.clone()
float_tensor = paddle.rand([128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_9_tensor = f16_tensor
arg_9 = arg_9_tensor.clone()
arg_10 = 0.0
arg_11 = 0.1
arg_12 = "relu"
arg_13 = 1e-05
arg_14 = 1e-05
arg_15 = 97
arg_16 = True
arg_17 = -1
arg_18 = None
start = time.time()
results["time_low"] = paddle.incubate.nn.functional.fused_feedforward(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,dropout1_rate=arg_10,dropout2_rate=arg_11,activation=arg_12,ln1_epsilon=arg_13,ln2_epsilon=arg_14,pre_layer_norm=arg_15,training=arg_16,ring_id=arg_17,name=arg_18,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_4 = arg_4_tensor.clone().type(paddle.float32)
arg_5 = arg_5_tensor.clone().type(paddle.float32)
arg_8 = arg_8_tensor.clone().type(paddle.float32)
arg_9 = arg_9_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.incubate.nn.functional.fused_feedforward(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,dropout1_rate=arg_10,dropout2_rate=arg_11,activation=arg_12,ln1_epsilon=arg_13,ln2_epsilon=arg_14,pre_layer_norm=arg_15,training=arg_16,ring_id=arg_17,name=arg_18,)
results["time_high"] = time.time() - start

print(results)

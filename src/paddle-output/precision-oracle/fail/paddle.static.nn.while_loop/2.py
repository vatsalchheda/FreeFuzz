results = dict()
import paddle
import time
arg_1 = "cond_zoom"
arg_2 = -1
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_0_tensor = int8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(0,2,[1])
arg_3_1 = arg_3_1_tensor.clone()
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_2_tensor = f16_tensor
arg_3_2 = arg_3_2_tensor.clone()
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_3_tensor = f16_tensor
arg_3_3 = arg_3_3_tensor.clone()
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_4_tensor = f16_tensor
arg_3_4 = arg_3_4_tensor.clone()
float_tensor = paddle.rand([2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_5_tensor = f16_tensor
arg_3_5 = arg_3_5_tensor.clone()
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_6_tensor = f16_tensor
arg_3_6 = arg_3_6_tensor.clone()
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_7_tensor = f16_tensor
arg_3_7 = arg_3_7_tensor.clone()
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_8_tensor = f16_tensor
arg_3_8 = arg_3_8_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,arg_3_7,arg_3_8,]
start = time.time()
results["time_low"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.int64)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.bool)
arg_3_2 = arg_3_2_tensor.clone().type(paddle.float32)
arg_3_3 = arg_3_3_tensor.clone().type(paddle.float32)
arg_3_4 = arg_3_4_tensor.clone().type(paddle.float32)
arg_3_5 = arg_3_5_tensor.clone().type(paddle.float32)
arg_3_6 = arg_3_6_tensor.clone().type(paddle.float32)
arg_3_7 = arg_3_7_tensor.clone().type(paddle.float32)
arg_3_8 = arg_3_8_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,arg_3_7,arg_3_8,]
start = time.time()
results["time_high"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
results["time_high"] = time.time() - start

print(results)

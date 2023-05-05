results = dict()
import paddle
import time
float_tensor = paddle.rand([16, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1024
arg_2_1 = 64
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 50.0
start = time.time()
results["time_low"] = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
results["time_high"] = time.time() - start

print(results)

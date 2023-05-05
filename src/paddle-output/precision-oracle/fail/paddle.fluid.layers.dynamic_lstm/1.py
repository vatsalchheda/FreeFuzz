results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([-1, 2048], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2048
arg_3 = -1
start = time.time()
results["time_low"] = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
results["time_high"] = time.time() - start

print(results)

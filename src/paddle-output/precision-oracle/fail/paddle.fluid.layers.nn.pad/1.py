results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([-1, -1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 37
arg_2_1 = 7
arg_2_2 = -47
arg_2_3 = -57
arg_2_4 = -4
arg_2_5 = -1024
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,arg_2_5,]
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.pad(arg_1,paddings=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,arg_2_5,]
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.pad(arg_1,paddings=arg_2,)
results["time_high"] = time.time() - start

print(results)

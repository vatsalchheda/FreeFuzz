results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,2,[2, 21, 10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -37
arg_2_1 = 1
arg_2_2 = 0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.transpose(arg_1,perm=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.transpose(arg_1,perm=arg_2,)
results["time_high"] = time.time() - start

print(results)

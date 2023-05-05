results = dict()
import paddle
import time
arg_1 = -62
arg_2 = 4
arg_class = paddle.fluid.dygraph.nn.GroupNorm(channels=arg_1,groups=arg_2,)
arg_3_0_tensor = paddle.rand([8, 32, 32], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

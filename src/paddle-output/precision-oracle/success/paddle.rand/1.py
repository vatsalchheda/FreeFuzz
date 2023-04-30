results = dict()
import paddle
import time
arg_1_0 = 48
arg_1 = [arg_1_0,]
start = time.time()
results["time_low"] = paddle.rand(arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.rand(arg_1,)
results["time_high"] = time.time() - start

print(results)

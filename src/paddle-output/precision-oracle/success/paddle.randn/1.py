results = dict()
import paddle
import time
arg_1_0 = 3
arg_1 = [arg_1_0,]
start = time.time()
results["time_low"] = paddle.randn(shape=arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.randn(shape=arg_1,)
results["time_high"] = time.time() - start

print(results)

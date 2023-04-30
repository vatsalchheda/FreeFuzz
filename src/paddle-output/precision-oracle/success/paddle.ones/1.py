results = dict()
import paddle
import time
arg_1_0 = 2
arg_1_1 = 2
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.ones(arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.ones(arg_1,)
results["time_high"] = time.time() - start

print(results)

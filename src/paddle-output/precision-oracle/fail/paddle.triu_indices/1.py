results = dict()
import paddle
import time
arg_1 = -23
arg_2 = 4
arg_3 = -72
start = time.time()
results["time_low"] = paddle.triu_indices(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.triu_indices(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

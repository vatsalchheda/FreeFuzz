results = dict()
import paddle
import time
arg_1 = 4
arg_2 = True
arg_3 = 0
start = time.time()
results["time_low"] = paddle.tril_indices(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.tril_indices(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

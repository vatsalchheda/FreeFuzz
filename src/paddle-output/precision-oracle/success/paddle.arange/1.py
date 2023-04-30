results = dict()
import paddle
import time
arg_1 = 23
arg_2 = "float32"
start = time.time()
results["time_low"] = paddle.arange(arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.arange(arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)

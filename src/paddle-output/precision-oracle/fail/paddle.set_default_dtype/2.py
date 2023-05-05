results = dict()
import paddle
import time
arg_1 = -1037.0
start = time.time()
results["time_low"] = paddle.set_default_dtype(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.set_default_dtype(arg_1,)
results["time_high"] = time.time() - start

print(results)

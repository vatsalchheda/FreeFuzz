results = dict()
import paddle
import time
arg_1 = "mean"
start = time.time()
results["time_low"] = paddle.static.device_guard(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.device_guard(arg_1,)
results["time_high"] = time.time() - start

print(results)

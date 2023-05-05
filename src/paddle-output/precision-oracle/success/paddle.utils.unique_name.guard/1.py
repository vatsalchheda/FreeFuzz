results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.utils.unique_name.guard()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.utils.unique_name.guard()
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.static.global_scope()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.global_scope()
results["time_high"] = time.time() - start

print(results)

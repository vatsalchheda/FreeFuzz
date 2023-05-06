results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.static.cpu_places()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.cpu_places()
results["time_high"] = time.time() - start

print(results)

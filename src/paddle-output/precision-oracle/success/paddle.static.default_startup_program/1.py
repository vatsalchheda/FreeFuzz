results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.static.default_startup_program()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.default_startup_program()
results["time_high"] = time.time() - start

print(results)

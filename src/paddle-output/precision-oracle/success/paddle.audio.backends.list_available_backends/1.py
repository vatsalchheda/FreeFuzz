results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.audio.backends.list_available_backends()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.backends.list_available_backends()
results["time_high"] = time.time() - start

print(results)

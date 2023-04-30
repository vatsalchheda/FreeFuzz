results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.device.get_available_custom_device()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.device.get_available_custom_device()
results["time_high"] = time.time() - start

print(results)

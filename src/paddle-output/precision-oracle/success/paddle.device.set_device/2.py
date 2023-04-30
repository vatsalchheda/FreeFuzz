results = dict()
import paddle
import time
arg_1 = "cpu"
start = time.time()
results["time_low"] = paddle.device.set_device(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.device.set_device(arg_1,)
results["time_high"] = time.time() - start

print(results)

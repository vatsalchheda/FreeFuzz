results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.device.is_compiled_with_npu()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.device.is_compiled_with_npu()
results["time_high"] = time.time() - start

print(results)

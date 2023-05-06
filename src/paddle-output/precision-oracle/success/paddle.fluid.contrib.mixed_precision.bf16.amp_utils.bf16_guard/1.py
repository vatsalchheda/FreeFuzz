results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.fluid.contrib.mixed_precision.bf16.amp_utils.bf16_guard()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.contrib.mixed_precision.bf16.amp_utils.bf16_guard()
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.fluid.framework.in_dygraph_mode()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.framework.in_dygraph_mode()
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_class = paddle.fluid.layers.control_flow.DynamicRNN()
arg_1 = []
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1 = []
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)

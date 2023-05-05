results = dict()
import paddle
import time
arg_1 = "paddleVarType"
start = time.time()
results["time_low"] = paddle.fluid.layers.control_flow.create_array(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.layers.control_flow.create_array(arg_1,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1 = None
start = time.time()
results["time_low"] = paddle.fluid.dygraph.base.guard(place=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.dygraph.base.guard(place=arg_1,)
results["time_high"] = time.time() - start

print(results)

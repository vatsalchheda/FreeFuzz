results = dict()
import paddle
import time
arg_1 = "builtinsbytes"
start = time.time()
results["time_low"] = paddle.static.deserialize_program(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.deserialize_program(arg_1,)
results["time_high"] = time.time() - start

print(results)

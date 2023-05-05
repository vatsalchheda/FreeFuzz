results = dict()
import paddle
import time
arg_1 = 1072
start = time.time()
results["time_low"] = paddle.jit.set_code_level(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.jit.set_code_level(arg_1,)
results["time_high"] = time.time() - start

print(results)

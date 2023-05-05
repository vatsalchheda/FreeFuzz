results = dict()
import paddle
import time
arg_1 = "regex"
start = time.time()
results["time_low"] = paddle.utils.try_import(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.utils.try_import(arg_1,)
results["time_high"] = time.time() - start

print(results)

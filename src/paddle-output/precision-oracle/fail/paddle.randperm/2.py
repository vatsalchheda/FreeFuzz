results = dict()
import paddle
import time
arg_1 = 40
arg_2 = "paddleVarType"
start = time.time()
results["time_low"] = paddle.randperm(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.randperm(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

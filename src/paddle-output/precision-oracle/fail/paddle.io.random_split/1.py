results = dict()
import paddle
import time
arg_1 = -36.0
arg_2_0 = -60.0
arg_2_1 = "zeros"
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.io.random_split(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.io.random_split(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

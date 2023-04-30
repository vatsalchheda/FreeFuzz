results = dict()
import paddle
import time
arg_1 = 0.75
arg_2 = 51.391925438912004
arg_3 = 19
arg_4 = 44
start = time.time()
results["time_low"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_high"] = time.time() - start

print(results)

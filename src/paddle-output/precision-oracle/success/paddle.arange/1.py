results = dict()
import paddle
import time
arg_1 = 0
arg_2 = 65
arg_3 = "float32"
arg_4 = None
start = time.time()
results["time_low"] = paddle.arange(arg_1,arg_2,dtype=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.arange(arg_1,arg_2,dtype=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)

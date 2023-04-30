results = dict()
import paddle
import time
arg_1_0 = 1024
arg_1_1 = 66
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
arg_3 = 0.0
arg_4 = 1.0
start = time.time()
results["time_low"] = paddle.uniform(arg_1,dtype=arg_2,min=arg_3,max=arg_4,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.uniform(arg_1,dtype=arg_2,min=arg_3,max=arg_4,)
results["time_high"] = time.time() - start

print(results)

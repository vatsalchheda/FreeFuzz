results = dict()
import paddle
import time
arg_1 = -1.0
arg_2 = -2.9800000000000004
arg_3_0 = -27
arg_3_1 = 66
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
results["time_high"] = time.time() - start

print(results)

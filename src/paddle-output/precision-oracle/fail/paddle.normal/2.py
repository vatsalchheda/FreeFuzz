results = dict()
import paddle
import time
arg_1 = 44.0
arg_2 = -46.98
arg_3_0 = "max"
arg_3_1 = -64.0
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1 = 0.0
arg_2 = "reflect"
arg_3_0 = 8
arg_3_1 = 8
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
results["time_high"] = time.time() - start

print(results)

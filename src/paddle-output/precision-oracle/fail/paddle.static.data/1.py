results = dict()
import paddle
import time
arg_1 = "pre_h"
arg_2_0 = 4
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "float64"
start = time.time()
results["time_low"] = paddle.static.data(arg_1,arg_2,dtype=arg_3,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.static.data(arg_1,arg_2,dtype=arg_3,)
results["time_high"] = time.time() - start

print(results)

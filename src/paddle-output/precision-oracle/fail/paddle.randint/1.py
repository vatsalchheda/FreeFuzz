results = dict()
import paddle
import time
arg_1 = 0
arg_2 = 83.0
arg_3_0 = 10
arg_3_1 = 4
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.randint(arg_1,arg_2,shape=arg_3,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.randint(arg_1,arg_2,shape=arg_3,)
results["time_high"] = time.time() - start

print(results)

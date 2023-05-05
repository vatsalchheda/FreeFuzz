results = dict()
import paddle
import time
arg_1 = 0
arg_2 = 5
arg_3_0 = -31
arg_3_1 = -31
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.randint(low=arg_1,high=arg_2,shape=arg_3,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.randint(low=arg_1,high=arg_2,shape=arg_3,)
results["time_high"] = time.time() - start

print(results)

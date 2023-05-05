results = dict()
import paddle
import time
arg_1 = "__main__LeNet"
arg_2_0 = 1
arg_2_1 = 1
arg_2_2 = 28
arg_2_3 = 28
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_low"] = paddle.summary(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.summary(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

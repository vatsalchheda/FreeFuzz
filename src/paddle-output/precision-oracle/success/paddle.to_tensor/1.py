results = dict()
import paddle
import time
arg_1_0 = -0.4
arg_1_1 = -0.2
arg_1_2 = 0.1
arg_1_3 = 0.3
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_low"] = paddle.to_tensor(arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_high"] = paddle.to_tensor(arg_1,)
results["time_high"] = time.time() - start

print(results)

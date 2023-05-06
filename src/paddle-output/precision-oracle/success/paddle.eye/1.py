results = dict()
import paddle
import time
arg_1 = 32
arg_2 = 57
arg_3 = "int32"
start = time.time()
results["time_low"] = paddle.eye(arg_1,arg_2,dtype=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.eye(arg_1,arg_2,dtype=arg_3,)
results["time_high"] = time.time() - start

print(results)

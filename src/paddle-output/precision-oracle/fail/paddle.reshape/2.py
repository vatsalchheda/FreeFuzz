results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,128,[6], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = False
start = time.time()
results["time_low"] = paddle.reshape(arg_1,arg_2,name=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.reshape(arg_1,arg_2,name=arg_3,)
results["time_high"] = time.time() - start

print(results)

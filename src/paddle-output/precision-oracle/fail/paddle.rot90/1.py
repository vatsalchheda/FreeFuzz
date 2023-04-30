results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,2,[2, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.rot90(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.rot90(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

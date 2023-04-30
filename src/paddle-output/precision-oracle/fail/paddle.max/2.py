results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,2,[2, 2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.max(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.max(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)

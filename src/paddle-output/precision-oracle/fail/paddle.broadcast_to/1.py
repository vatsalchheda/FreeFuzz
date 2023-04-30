results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,1,[1], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = paddle.broadcast_to(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.broadcast_to(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

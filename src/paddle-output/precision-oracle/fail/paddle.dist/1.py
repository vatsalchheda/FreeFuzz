results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,8192,[2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,8,[2, 2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = 9.5
start = time.time()
results["time_low"] = paddle.dist(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.dist(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

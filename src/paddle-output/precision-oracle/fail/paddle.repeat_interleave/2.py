results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,1,[2, 3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3 = None
start = time.time()
results["time_low"] = paddle.repeat_interleave(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.repeat_interleave(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

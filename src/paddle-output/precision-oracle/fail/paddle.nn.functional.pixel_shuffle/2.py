results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,1,[5], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
start = time.time()
results["time_low"] = paddle.nn.functional.pixel_shuffle(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.pixel_shuffle(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

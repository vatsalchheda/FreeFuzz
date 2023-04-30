results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,1,[3, 3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,4,[1, 1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
start = time.time()
results["time_low"] = paddle.take_along_axis(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.take_along_axis(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

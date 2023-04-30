results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4096,4096,[4, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,1,[4, 1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 34
start = time.time()
results["time_low"] = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)

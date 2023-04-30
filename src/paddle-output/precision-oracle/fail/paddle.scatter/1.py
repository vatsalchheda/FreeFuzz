results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,512,[3, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,4,[4], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-512,32768,[9, 2], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = True
start = time.time()
results["time_low"] = paddle.scatter(arg_1,arg_2,arg_3,overwrite=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.scatter(arg_1,arg_2,arg_3,overwrite=arg_4,)
results["time_high"] = time.time() - start

print(results)

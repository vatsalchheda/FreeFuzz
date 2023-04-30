results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2048,256,[2, 2, 1, 1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,8,[1], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.greater_equal(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.greater_equal(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

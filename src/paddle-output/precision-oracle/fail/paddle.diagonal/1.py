results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,8,[2, 2, 3, 1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.diagonal(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.diagonal(arg_1,)
results["time_high"] = time.time() - start

print(results)

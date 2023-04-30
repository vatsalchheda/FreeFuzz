results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,256,[4, 4], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.angle(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex64)
start = time.time()
results["time_high"] = paddle.angle(arg_1,)
results["time_high"] = time.time() - start

print(results)

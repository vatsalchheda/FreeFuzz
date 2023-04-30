results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,16,[4, 1], dtype=paddle.uint8)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.erf(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.uint8)
start = time.time()
results["time_high"] = paddle.erf(arg_1,)
results["time_high"] = time.time() - start

print(results)

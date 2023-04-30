results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4096,32768,[4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.67
arg_3 = -13.28
start = time.time()
results["time_low"] = paddle.stanh(arg_1,scale_a=arg_2,scale_b=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.stanh(arg_1,scale_a=arg_2,scale_b=arg_3,)
results["time_high"] = time.time() - start

print(results)

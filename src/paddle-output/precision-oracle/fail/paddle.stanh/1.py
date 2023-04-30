results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,1,[4, 12], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = -18.33
arg_3 = 1.72
start = time.time()
results["time_low"] = paddle.stanh(arg_1,scale_a=arg_2,scale_b=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.stanh(arg_1,scale_a=arg_2,scale_b=arg_3,)
results["time_high"] = time.time() - start

print(results)

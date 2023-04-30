results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32,2048,[3, 9, 5], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 45
arg_3 = -2
start = time.time()
results["time_low"] = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)

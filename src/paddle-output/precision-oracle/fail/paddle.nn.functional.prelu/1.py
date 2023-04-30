results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,1,[1, 2, 3, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,2048,[1], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.nn.functional.prelu(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.prelu(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

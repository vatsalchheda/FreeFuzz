results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4096,1024,[-1, 6, 26, 26], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = "relu"
start = time.time()
results["time_low"] = paddle.static.nn.batch_norm(input=arg_1,act=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.nn.batch_norm(input=arg_1,act=arg_2,)
results["time_high"] = time.time() - start

print(results)

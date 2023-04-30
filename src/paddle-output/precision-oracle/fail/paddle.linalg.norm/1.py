results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4,128,[2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = inf
start = time.time()
results["time_low"] = paddle.linalg.norm(arg_1,p=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.linalg.norm(arg_1,p=arg_2,)
results["time_high"] = time.time() - start

print(results)

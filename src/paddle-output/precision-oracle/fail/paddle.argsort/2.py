results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,8192,[64, 10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = True
start = time.time()
results["time_low"] = paddle.argsort(arg_1,descending=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.argsort(arg_1,descending=arg_2,)
results["time_high"] = time.time() - start

print(results)

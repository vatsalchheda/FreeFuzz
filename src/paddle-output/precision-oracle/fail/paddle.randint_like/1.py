results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -0.17677669529663687
arg_3 = 5
start = time.time()
results["time_low"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
results["time_high"] = time.time() - start

print(results)

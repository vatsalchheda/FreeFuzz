results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,2,[3, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
start = time.time()
results["time_low"] = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
results["time_high"] = time.time() - start

print(results)

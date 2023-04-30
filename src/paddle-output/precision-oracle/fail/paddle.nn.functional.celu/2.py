results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2048,32,[1, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.2
arg_3 = None
start = time.time()
results["time_low"] = paddle.nn.functional.celu(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.nn.functional.celu(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

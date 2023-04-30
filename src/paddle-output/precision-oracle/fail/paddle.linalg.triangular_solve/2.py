results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,2048,[3, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,128,[1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = "sum"
start = time.time()
results["time_low"] = paddle.linalg.triangular_solve(arg_1,arg_2,upper=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.linalg.triangular_solve(arg_1,arg_2,upper=arg_3,)
results["time_high"] = time.time() - start

print(results)

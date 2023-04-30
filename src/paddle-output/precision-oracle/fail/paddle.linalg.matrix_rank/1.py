results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,128,[2, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = -8.99
arg_3 = "max"
start = time.time()
results["time_low"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
results["time_high"] = time.time() - start

print(results)

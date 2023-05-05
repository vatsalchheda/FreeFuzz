results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 4, 5, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 0.01
arg_3 = True
start = time.time()
results["time_low"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = False
start = time.time()
results["time_low"] = paddle.linalg.cholesky_solve(arg_1,arg_2,upper=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.linalg.cholesky_solve(arg_1,arg_2,upper=arg_3,)
results["time_high"] = time.time() - start

print(results)

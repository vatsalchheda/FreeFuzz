results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = inf
start = time.time()
results["time_low"] = paddle.linalg.norm(arg_1,p=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.linalg.norm(arg_1,p=arg_2,)
results["time_high"] = time.time() - start

print(results)

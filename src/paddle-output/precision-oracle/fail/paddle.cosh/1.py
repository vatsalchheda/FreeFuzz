results = dict()
import paddle
import time
float_tensor = paddle.rand([4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.cosh(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.cosh(arg_1,)
results["time_high"] = time.time() - start

print(results)

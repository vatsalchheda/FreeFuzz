results = dict()
import paddle
import time
arg_1 = -34.0
float_tensor = paddle.rand([2, 3, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([2, 2, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.einsum(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.einsum(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

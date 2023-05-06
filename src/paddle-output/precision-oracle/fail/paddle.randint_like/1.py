results = dict()
import paddle
import time
float_tensor = paddle.rand([128, 128, 11, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = -5
arg_3 = 5
start = time.time()
results["time_low"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
results["time_high"] = time.time() - start

print(results)

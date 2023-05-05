results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = -1
start = time.time()
results["time_low"] = paddle.amax(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.amax(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)

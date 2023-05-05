results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 1536], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 4
arg_3 = -63.0
arg_4 = 1.0
start = time.time()
results["time_low"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
results["time_high"] = time.time() - start

print(results)

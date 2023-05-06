results = dict()
import paddle
import time
float_tensor = paddle.rand([16, 96], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = -21.0
start = time.time()
results["time_low"] = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)

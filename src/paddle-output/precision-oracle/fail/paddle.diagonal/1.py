results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 2, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 49.0
arg_3 = 2
arg_4 = 1038.0
start = time.time()
results["time_low"] = paddle.diagonal(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.diagonal(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
results["time_high"] = time.time() - start

print(results)

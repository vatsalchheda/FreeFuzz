results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_0_tensor = f16_tensor
arg_1_0 = arg_1_0_tensor.clone()
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_1_tensor = f16_tensor
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_0_tensor = f16_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = True
start = time.time()
results["time_low"] = paddle.autograd.backward(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.autograd.backward(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

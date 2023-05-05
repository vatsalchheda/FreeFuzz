results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 23, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_0_tensor = f16_tensor
arg_1_0 = arg_1_0_tensor.clone()
float_tensor = paddle.rand([2, 23, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_1_tensor = f16_tensor
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = -999
start = time.time()
results["time_low"] = paddle.fluid.layers.tensor.concat(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.fluid.layers.tensor.concat(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

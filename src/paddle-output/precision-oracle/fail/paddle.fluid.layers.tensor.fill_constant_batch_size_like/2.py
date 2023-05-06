results = dict()
import paddle
import time
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2 = [arg_2_0,]
arg_3 = "paddleVarType"
arg_4 = 0.0
start = time.time()
results["time_low"] = paddle.fluid.layers.tensor.fill_constant_batch_size_like(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.fluid.layers.tensor.fill_constant_batch_size_like(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

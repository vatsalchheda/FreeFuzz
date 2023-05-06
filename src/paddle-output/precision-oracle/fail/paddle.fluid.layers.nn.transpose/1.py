results = dict()
import paddle
import time
float_tensor = paddle.rand([4, 12], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 60
arg_2_1 = -10
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.transpose(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.transpose(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

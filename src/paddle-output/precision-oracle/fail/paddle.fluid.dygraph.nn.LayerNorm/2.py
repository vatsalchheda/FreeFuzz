results = dict()
import paddle
import time
arg_1_0 = 32
arg_1_1 = 32
arg_1 = [arg_1_0,arg_1_1,]
arg_class = paddle.fluid.dygraph.nn.LayerNorm(arg_1,)
float_tensor = paddle.rand([3, 32, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_0_tensor = f16_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

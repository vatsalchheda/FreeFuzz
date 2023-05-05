results = dict()
import paddle
import time
arg_1 = 3
arg_2 = 2
arg_3 = 3
arg_class = paddle.fluid.dygraph.nn.Conv2D(arg_1,arg_2,arg_3,)
float_tensor = paddle.rand([10, 3, 32, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_0_tensor = f16_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

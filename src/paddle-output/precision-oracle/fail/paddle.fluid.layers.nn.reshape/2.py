results = dict()
import paddle
import time
float_tensor = paddle.rand([32, 32, 7, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 32
arg_2_1 = 32
arg_2_2 = 7
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = -92
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.reshape(arg_1,arg_2,name=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.reshape(arg_1,arg_2,name=arg_3,)
results["time_high"] = time.time() - start

print(results)

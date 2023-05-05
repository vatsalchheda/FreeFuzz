results = dict()
import paddle
import time
arg_1 = None
arg_2 = -13.76393202250021
arg_3 = "leaky_relu"
arg_class = paddle.nn.initializer.KaimingUniform(fan_in=arg_1,negative_slope=arg_2,nonlinearity=arg_3,)
float_tensor = paddle.rand([2048], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_0_tensor = f16_tensor
arg_4_0 = arg_4_0_tensor.clone()
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_1_tensor = f16_tensor
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

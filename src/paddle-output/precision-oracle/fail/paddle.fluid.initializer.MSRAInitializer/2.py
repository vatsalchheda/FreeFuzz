results = dict()
import paddle
import time
arg_1 = True
arg_2 = None
arg_3 = -5
arg_4 = 11.23606797749979
arg_5 = "leaky_relu"
arg_class = paddle.fluid.initializer.MSRAInitializer(uniform=arg_1,fan_in=arg_2,seed=arg_3,negative_slope=arg_4,nonlinearity=arg_5,)
float_tensor = paddle.rand([512], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_6_0_tensor = f16_tensor
arg_6_0 = arg_6_0_tensor.clone()
real = paddle.rand([2, 0], paddle.float32)
imag = paddle.rand([2, 0], paddle.float32)
arg_6_1_tensor = paddle.complex(real, imag)
arg_6_1 = arg_6_1_tensor.clone()
arg_6 = [arg_6_0,arg_6_1,]
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
arg_6_0 = arg_6_0_tensor.clone().type(paddle.float64)
arg_6_1 = arg_6_1_tensor.clone().type(paddle.complex128)
arg_6 = [arg_6_0,arg_6_1,]
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)

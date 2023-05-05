results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_class = paddle.nn.initializer.Assign(arg_1,)
real = paddle.rand([1], paddle.float32)
imag = paddle.rand([1], paddle.float32)
arg_2_0_tensor = paddle.complex(real, imag)
arg_2_0 = arg_2_0_tensor.clone()
float_tensor = paddle.rand([37], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_1_tensor = f16_tensor
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2_0 = arg_2_0_tensor.clone().type(paddle.complex64)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

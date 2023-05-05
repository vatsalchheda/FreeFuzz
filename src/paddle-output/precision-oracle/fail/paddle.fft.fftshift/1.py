results = dict()
import paddle
import time
real = paddle.rand([5, 5], paddle.float32)
imag = paddle.rand([5, 5], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = paddle.fft.fftshift(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.fft.fftshift(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

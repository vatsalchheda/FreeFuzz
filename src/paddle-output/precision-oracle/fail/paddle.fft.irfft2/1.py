results = dict()
import paddle
import time
real = paddle.rand([4, 0, 4], paddle.float32)
imag = paddle.rand([4, 0, 4], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = False
arg_3_1 = -5.0
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.irfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.complex128)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.irfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

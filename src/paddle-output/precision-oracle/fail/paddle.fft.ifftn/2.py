results = dict()
import paddle
import time
real = paddle.rand([3, 3, 7, 5, 4], paddle.float32)
imag = paddle.rand([3, 3, 7, 5, 4], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -36
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

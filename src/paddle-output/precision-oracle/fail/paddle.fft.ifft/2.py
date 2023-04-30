results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,1,[5, 5, 8, 5, 6], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = 19.0
arg_4 = "ortho"
start = time.time()
results["time_low"] = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,4,[4, 4, 4], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -23
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.hfft(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.fft.hfft(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

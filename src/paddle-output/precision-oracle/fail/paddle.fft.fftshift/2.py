results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,1024,[10, 10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = None
start = time.time()
results["time_low"] = paddle.fft.fftshift(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.fft.fftshift(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

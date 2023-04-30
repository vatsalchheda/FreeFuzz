results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,16,[4, 4, 4], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = "max"
arg_3_1 = "max"
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "forward"
start = time.time()
results["time_low"] = paddle.fft.irfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.irfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

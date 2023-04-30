results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,2,[6, 9, 5, 9, 7], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 127.0
arg_3_1 = 1075.0
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "ortho"
start = time.time()
results["time_low"] = paddle.fft.ihfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.ihfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

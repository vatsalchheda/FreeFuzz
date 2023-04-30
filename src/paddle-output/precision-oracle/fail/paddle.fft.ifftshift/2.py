results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,64,[10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 1
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.fft.ifftshift(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.fft.ifftshift(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32,2,[5, 48, 9, 9, 7], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = 3
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

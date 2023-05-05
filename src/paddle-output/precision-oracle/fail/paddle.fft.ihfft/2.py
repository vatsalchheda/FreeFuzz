results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 1, 3, 2, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 11
arg_3 = -6
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.ihfft(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.fft.ihfft(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

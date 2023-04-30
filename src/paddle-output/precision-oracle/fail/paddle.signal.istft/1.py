results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,8192,[8, 257, 376], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
start = time.time()
results["time_low"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
results["time_high"] = time.time() - start

print(results)

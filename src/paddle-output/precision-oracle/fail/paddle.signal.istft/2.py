results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[1, 1])
arg_1 = arg_1_tensor.clone()
arg_2 = 501
start = time.time()
results["time_low"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.bool)
start = time.time()
results["time_high"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
results["time_high"] = time.time() - start

print(results)

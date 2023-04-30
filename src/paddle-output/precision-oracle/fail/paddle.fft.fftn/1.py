results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,8,[4, 4, 4], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 38
arg_2_1 = 12
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.fft.fftn(arg_1,axes=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.fft.fftn(arg_1,axes=arg_2,)
results["time_high"] = time.time() - start

print(results)

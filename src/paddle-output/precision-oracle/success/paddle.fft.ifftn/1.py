results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([4, 4, 3, 4, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 6
arg_2_1 = 6
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 1
arg_3_1 = 2
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

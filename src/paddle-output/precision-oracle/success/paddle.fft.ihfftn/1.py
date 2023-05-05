results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([6, 2, 6, 9, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 1
arg_3_1 = 2
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

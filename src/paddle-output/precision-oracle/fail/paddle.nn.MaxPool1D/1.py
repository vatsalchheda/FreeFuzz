results = dict()
import paddle
import time
arg_1 = 2
arg_2 = -25
arg_3 = -11
arg_4 = True
arg_class = paddle.nn.MaxPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,return_mask=arg_4,)
arg_5_0_tensor = paddle.rand([17, 3], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().astype(paddle.float32)
arg_5 = [arg_5_0,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

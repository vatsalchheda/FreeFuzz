results = dict()
import paddle
import time
arg_1 = False
arg_2 = -5
arg_3 = 0
arg_class = paddle.nn.AvgPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4_0_tensor = paddle.rand([1, 2, 3, 32, 0], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.float32)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

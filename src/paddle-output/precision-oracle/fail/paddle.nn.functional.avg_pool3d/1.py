results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 1, 7, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 61.0
arg_2_1 = 62.0
arg_2_2 = False
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = -13
start = time.time()
results["time_low"] = paddle.nn.functional.avg_pool3d(arg_1,kernel_size=arg_2,stride=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.nn.functional.avg_pool3d(arg_1,kernel_size=arg_2,stride=arg_3,)
results["time_high"] = time.time() - start

print(results)

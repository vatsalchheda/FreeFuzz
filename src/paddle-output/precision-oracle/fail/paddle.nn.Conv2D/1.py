results = dict()
import paddle
import time
arg_1 = 6
arg_2 = 19
arg_3 = 5
arg_4 = -16
arg_5 = 0
arg_class = paddle.nn.Conv2D(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,)
arg_6_0_tensor = paddle.rand([64, 6, 14, 14], dtype=paddle.float32)
arg_6_0 = arg_6_0_tensor.clone()
arg_6 = [arg_6_0,]
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
arg_6_0 = arg_6_0_tensor.clone().astype(paddle.float32)
arg_6 = [arg_6_0,]
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)

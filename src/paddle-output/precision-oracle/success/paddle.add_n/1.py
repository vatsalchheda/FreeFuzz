results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
start = time.time()
results["time_low"] = paddle.add_n(arg_1,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.float32)
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.add_n(arg_1,)
results["time_high"] = time.time() - start

print(results)

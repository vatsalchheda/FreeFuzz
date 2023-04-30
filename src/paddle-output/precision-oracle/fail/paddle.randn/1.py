results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-1,8,[1], dtype=paddle.int8)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8,2,[1], dtype=paddle.int8)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.randn(shape=arg_1,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.int64)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.int32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.randn(shape=arg_1,)
results["time_high"] = time.time() - start

print(results)

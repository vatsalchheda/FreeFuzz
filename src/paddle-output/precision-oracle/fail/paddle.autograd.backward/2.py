results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-4,4096,[2, 2], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8192,8,[2, 2], dtype=paddle.float16)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.autograd.backward(arg_1,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.autograd.backward(arg_1,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-512,8,[32, 1], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2 = None
start = time.time()
results["time_low"] = paddle.nn.functional.mish(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.nn.functional.mish(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

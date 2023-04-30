results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,32768,[3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = None
start = time.time()
results["time_low"] = paddle.nn.functional.hardshrink(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.hardshrink(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

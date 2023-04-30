results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,64,[2, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
start = time.time()
results["time_low"] = paddle.diagflat(arg_1,offset=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.diagflat(arg_1,offset=arg_2,)
results["time_high"] = time.time() - start

print(results)

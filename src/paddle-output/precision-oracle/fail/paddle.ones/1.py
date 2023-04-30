results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,1,[2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = "int32"
start = time.time()
results["time_low"] = paddle.ones(shape=arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.ones(shape=arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)

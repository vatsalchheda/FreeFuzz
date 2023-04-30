results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,2,[13], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 8
arg_3 = 2
start = time.time()
results["time_low"] = paddle.tril_indices(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.tril_indices(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

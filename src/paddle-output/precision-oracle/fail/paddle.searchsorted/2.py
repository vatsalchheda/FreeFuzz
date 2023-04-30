results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,32,[2, 6], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,4,[2, 4], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = True
start = time.time()
results["time_low"] = paddle.searchsorted(arg_1,arg_2,right=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.searchsorted(arg_1,arg_2,right=arg_3,)
results["time_high"] = time.time() - start

print(results)

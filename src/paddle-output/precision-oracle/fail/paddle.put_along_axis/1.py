results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,32,[2, 3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,64,[1, 1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 83
arg_4 = 0
start = time.time()
results["time_low"] = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

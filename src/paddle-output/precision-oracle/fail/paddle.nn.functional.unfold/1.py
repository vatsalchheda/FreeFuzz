results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,1,[2, 3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = -63
arg_4 = 1
arg_5 = 53
start = time.time()
results["time_low"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,8,[3, 4], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 0
arg_4 = False
start = time.time()
results["time_low"] = paddle.topk(arg_1,arg_2,axis=arg_3,largest=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.topk(arg_1,arg_2,axis=arg_3,largest=arg_4,)
results["time_high"] = time.time() - start

print(results)

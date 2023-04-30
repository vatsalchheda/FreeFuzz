results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,16384,[2, 3, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 1
start = time.time()
results["time_low"] = paddle.kthvalue(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.kthvalue(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

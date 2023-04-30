results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[1], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
start = time.time()
results["time_low"] = paddle.static.nn.cond(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.bool)
start = time.time()
results["time_high"] = paddle.static.nn.cond(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,256,[3, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[3, 4], dtype=paddle.bool)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.masked_select(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.bool)
start = time.time()
results["time_high"] = paddle.masked_select(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

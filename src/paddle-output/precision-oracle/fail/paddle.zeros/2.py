results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-128,64,[1], dtype=paddle.int8)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2 = "float64"
start = time.time()
results["time_low"] = paddle.zeros(arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.int64)
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.zeros(arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[64, 1])
arg_1 = arg_1_tensor.clone()
arg_2 = "float32"
start = time.time()
results["time_low"] = paddle.cast(arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.bool)
start = time.time()
results["time_high"] = paddle.cast(arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)

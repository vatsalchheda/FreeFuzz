results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[])
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.all(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.bool)
start = time.time()
results["time_high"] = paddle.all(arg_1,)
results["time_high"] = time.time() - start

print(results)

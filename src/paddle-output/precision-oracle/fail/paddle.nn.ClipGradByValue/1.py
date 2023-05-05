results = dict()
import paddle
import time
arg_1 = -1
arg_2 = "max"
arg_class = paddle.nn.ClipGradByValue(min=arg_1,max=arg_2,)
arg_3_0_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

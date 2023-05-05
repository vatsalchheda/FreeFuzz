results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1, 64, 13600], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = None
start = time.time()
results["time_low"] = paddle.nn.functional.leaky_relu(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.leaky_relu(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

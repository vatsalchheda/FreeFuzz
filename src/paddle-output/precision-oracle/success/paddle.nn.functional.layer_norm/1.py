results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1, 164, 1024], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 164
arg_2_1 = 1024
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.nn.functional.layer_norm(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.layer_norm(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

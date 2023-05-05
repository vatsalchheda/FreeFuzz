results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([10, 5], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([5, 8], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.rand([8, 7], dtype=paddle.float32)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_low"] = paddle.linalg.multi_dot(arg_1,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().astype(paddle.float32)
arg_1_2 = arg_1_2_tensor.clone().astype(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_high"] = paddle.linalg.multi_dot(arg_1,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.broadcast_tensors(arg_1,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().astype(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.broadcast_tensors(arg_1,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = True
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.reduce_max(arg_1,dim=arg_2,keep_dim=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.reduce_max(arg_1,dim=arg_2,keep_dim=arg_3,)
results["time_high"] = time.time() - start

print(results)

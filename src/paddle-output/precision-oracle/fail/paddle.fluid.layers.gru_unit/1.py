results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([-1, 30], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -10
start = time.time()
results["time_low"] = paddle.fluid.layers.gru_unit(input=arg_1,hidden=arg_2,size=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.gru_unit(input=arg_1,hidden=arg_2,size=arg_3,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,1024,[2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,8192,[3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.fluid.layers.loss.square_error_cost(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.loss.square_error_cost(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

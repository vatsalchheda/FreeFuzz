results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([5, 20], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([0, 20, 1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
start = time.time()
results["time_low"] = paddle.nn.functional.kl_div(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.float16)
start = time.time()
results["time_high"] = paddle.nn.functional.kl_div(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

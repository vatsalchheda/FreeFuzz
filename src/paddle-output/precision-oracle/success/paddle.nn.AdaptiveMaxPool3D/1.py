results = dict()
import paddle
import time
arg_1 = 28
arg_class = paddle.nn.AdaptiveMaxPool3D(output_size=arg_1,)
arg_2_0_tensor = paddle.rand([2, 3, 8, 32, 32], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

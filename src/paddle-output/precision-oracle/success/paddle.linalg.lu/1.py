results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
start = time.time()
results["time_low"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
results["time_high"] = time.time() - start

print(results)

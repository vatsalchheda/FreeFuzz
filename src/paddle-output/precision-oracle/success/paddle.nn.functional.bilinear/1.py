results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([5, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([5, 4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([1000, 5, 4], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.rand([1, 1000], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5 = None
start = time.time()
results["time_low"] = paddle.nn.functional.bilinear(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
arg_3 = arg_3_tensor.clone().astype(paddle.float32)
arg_4 = arg_4_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.bilinear(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)

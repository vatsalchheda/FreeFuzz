results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -5
arg_3 = 0.0001
arg_4 = 0.75
arg_5 = -56.0
arg_6 = "NCHW"
arg_7 = None
start = time.time()
results["time_low"] = paddle.nn.functional.local_response_norm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.local_response_norm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
results["time_high"] = time.time() - start

print(results)

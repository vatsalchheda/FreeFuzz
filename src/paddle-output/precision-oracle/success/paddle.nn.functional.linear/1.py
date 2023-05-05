results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([128, 128], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = None
start = time.time()
results["time_low"] = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
arg_3 = arg_3_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)

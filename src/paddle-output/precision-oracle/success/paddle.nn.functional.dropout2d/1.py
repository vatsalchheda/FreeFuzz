results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 2, 1, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = True
arg_4 = "NCHW"
arg_5 = None
start = time.time()
results["time_low"] = paddle.nn.functional.dropout2d(arg_1,p=arg_2,training=arg_3,data_format=arg_4,name=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.dropout2d(arg_1,p=arg_2,training=arg_3,data_format=arg_4,name=arg_5,)
results["time_high"] = time.time() - start

print(results)

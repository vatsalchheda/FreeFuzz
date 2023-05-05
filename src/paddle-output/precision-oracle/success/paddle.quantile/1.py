results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.8
arg_3 = 1
arg_4 = True
start = time.time()
results["time_low"] = paddle.quantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.quantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
results["time_high"] = time.time() - start

print(results)

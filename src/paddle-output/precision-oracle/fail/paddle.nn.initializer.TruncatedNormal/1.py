results = dict()
import paddle
import time
arg_1 = -1024.0
arg_2 = 0.02
arg_class = paddle.nn.initializer.TruncatedNormal(mean=arg_1,std=arg_2,)
arg_3_0_tensor = paddle.rand([768, 3072], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float32)
arg_3_1 = arg_3_1_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

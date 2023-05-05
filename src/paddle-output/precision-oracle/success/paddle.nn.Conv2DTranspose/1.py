results = dict()
import paddle
import time
arg_1 = 4
arg_2 = 32
arg_3_0 = 3
arg_3_1 = 3
arg_3 = [arg_3_0,arg_3_1,]
arg_class = paddle.nn.Conv2DTranspose(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([2, 4, 8, 8], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.float32)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_0 = 1
arg_1_1 = 0
arg_1_2 = 1
arg_1_3 = 2
arg_1_4 = 0
arg_1_5 = 0
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,arg_1_5,]
arg_2 = False
arg_class = paddle.nn.Pad3D(padding=arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.rand([1, 1, 1, 2, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,arg_1_5,]
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

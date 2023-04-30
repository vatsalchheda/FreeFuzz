results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-256,16,[100, 3, 224, 224], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -12
arg_2_1 = -39
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 1
arg_4 = -59
arg_5 = 1
start = time.time()
results["time_low"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)

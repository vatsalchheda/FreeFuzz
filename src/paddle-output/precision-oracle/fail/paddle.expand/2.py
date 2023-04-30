results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,512,[2, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.expand(arg_1,shape=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.expand(arg_1,shape=arg_2,)
results["time_high"] = time.time() - start

print(results)

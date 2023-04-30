results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32,16384,[4, 4, 4], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
arg_3_0 = 0
arg_3 = [arg_3_0,]
arg_4_0 = 3
arg_4_1 = 2
arg_4_2 = 4
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
start = time.time()
results["time_low"] = paddle.slice(arg_1,arg_2,starts=arg_3,ends=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
arg_2 = [arg_2_0,]
arg_3 = [arg_3_0,]
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
start = time.time()
results["time_high"] = paddle.slice(arg_1,arg_2,starts=arg_3,ends=arg_4,)
results["time_high"] = time.time() - start

print(results)

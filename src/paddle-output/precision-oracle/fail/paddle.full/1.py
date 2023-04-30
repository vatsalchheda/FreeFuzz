results = dict()
import paddle
import time
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2_tensor = paddle.randint(-128,1,[1], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = "paddleVarType"
start = time.time()
results["time_low"] = paddle.full(shape=arg_1,fill_value=arg_2,dtype=arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.full(shape=arg_1,fill_value=arg_2,dtype=arg_3,)
results["time_high"] = time.time() - start

print(results)

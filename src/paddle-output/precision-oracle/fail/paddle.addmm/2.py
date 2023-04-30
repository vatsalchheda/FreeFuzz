results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,256,[2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,16384,[2, 2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4,16384,[2, 2], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = 19.5
arg_5 = 5.0
start = time.time()
results["time_low"] = paddle.addmm(input=arg_1,x=arg_2,y=arg_3,beta=arg_4,alpha=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.addmm(input=arg_1,x=arg_2,y=arg_3,beta=arg_4,alpha=arg_5,)
results["time_high"] = time.time() - start

print(results)

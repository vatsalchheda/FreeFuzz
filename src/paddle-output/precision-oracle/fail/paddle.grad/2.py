results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,128,[2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,64,[2, 2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1,2048,[2, 2], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = False
arg_5 = True
start = time.time()
results["time_low"] = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)
results["time_high"] = time.time() - start

print(results)

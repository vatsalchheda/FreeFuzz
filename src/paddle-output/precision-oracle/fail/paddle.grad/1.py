results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,32,[1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-1024,2,[2], dtype=paddle.float16)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = False
start = time.time()
results["time_low"] = paddle.grad(arg_1,arg_2,create_graph=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.grad(arg_1,arg_2,create_graph=arg_3,)
results["time_high"] = time.time() - start

print(results)

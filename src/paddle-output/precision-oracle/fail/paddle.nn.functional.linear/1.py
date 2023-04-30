results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,1,[-1, 2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,8192,[2, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16384,16384,[3], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,)
results["time_high"] = time.time() - start

print(results)

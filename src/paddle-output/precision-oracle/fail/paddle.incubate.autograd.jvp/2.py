results = dict()
import paddle
import time
arg_1 = "func"
arg_2_tensor = paddle.randint(-32,4096,[2, 2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2,16,[2, 2], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.incubate.autograd.jvp(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.incubate.autograd.jvp(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

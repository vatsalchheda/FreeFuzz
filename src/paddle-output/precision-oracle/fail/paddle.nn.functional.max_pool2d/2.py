results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,16,[-1, 20, 24, 24], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = -18
arg_3 = 2
start = time.time()
results["time_low"] = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,)
results["time_high"] = time.time() - start

print(results)

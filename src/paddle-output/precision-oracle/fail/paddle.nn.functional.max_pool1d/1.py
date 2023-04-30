results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,4096,[1, 3, 32], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = "max"
arg_4 = 0
arg_5 = True
start = time.time()
results["time_low"] = paddle.nn.functional.max_pool1d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.max_pool1d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,)
results["time_high"] = time.time() - start

print(results)

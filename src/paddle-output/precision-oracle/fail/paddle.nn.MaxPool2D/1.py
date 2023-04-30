results = dict()
import paddle
import time
arg_1 = -47
arg_2 = 2
arg_3 = 0
arg_4 = True
arg_class = paddle.nn.MaxPool2D(kernel_size=arg_1,stride=arg_2,padding=arg_3,return_mask=arg_4,)
arg_5_0_tensor = paddle.randint(-8192,2,[1, 3, 32, 32], dtype=paddle.float16)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().type(paddle.float32)
arg_5 = [arg_5_0,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1 = 2
arg_2 = False
arg_3 = 0
arg_class = paddle.nn.MaxPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4_0_tensor = paddle.randint(-128,2048,[1, 2, 3, 32, 84, 31], dtype=paddle.bfloat16)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.bfloat16)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

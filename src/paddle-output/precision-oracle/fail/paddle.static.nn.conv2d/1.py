results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4096,8,[-1, 20, 12, 12], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3 = 20
arg_4 = "relu"
start = time.time()
results["time_low"] = paddle.static.nn.conv2d(input=arg_1,filter_size=arg_2,num_filters=arg_3,act=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.nn.conv2d(input=arg_1,filter_size=arg_2,num_filters=arg_3,act=arg_4,)
results["time_high"] = time.time() - start

print(results)

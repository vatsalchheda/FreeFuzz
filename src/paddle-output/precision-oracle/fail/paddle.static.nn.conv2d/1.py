results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 1, 28, 28], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 6
arg_3 = 3
start = time.time()
results["time_low"] = paddle.static.nn.conv2d(input=arg_1,num_filters=arg_2,filter_size=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.nn.conv2d(input=arg_1,num_filters=arg_2,filter_size=arg_3,)
results["time_high"] = time.time() - start

print(results)

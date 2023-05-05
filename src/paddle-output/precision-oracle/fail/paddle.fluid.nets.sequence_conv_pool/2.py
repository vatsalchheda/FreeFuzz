results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = "mean"
arg_3 = 3
arg_4 = "mean"
arg_5 = "sqrt"
start = time.time()
results["time_low"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
results["time_high"] = time.time() - start

print(results)

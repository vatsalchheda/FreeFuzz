results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 454
arg_3 = 21
arg_4 = "tanh"
arg_5 = 66.0
start = time.time()
results["time_low"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
results["time_high"] = time.time() - start

print(results)

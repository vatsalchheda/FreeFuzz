results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,2048,[-1, 128], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = -1024
arg_3 = 3
arg_4 = "tanh"
arg_5 = "sqrt"
start = time.time()
results["time_low"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
results["time_high"] = time.time() - start

print(results)

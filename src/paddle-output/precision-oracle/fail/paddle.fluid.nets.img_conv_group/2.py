results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 1, 28, 28], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 42
arg_3_0 = 3
arg_3_1 = 3
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = 3
arg_5 = "relu"
arg_6 = 20.0
arg_7 = 2
start = time.time()
results["time_low"] = paddle.fluid.nets.img_conv_group(input=arg_1,conv_padding=arg_2,conv_num_filter=arg_3,conv_filter_size=arg_4,conv_act=arg_5,pool_size=arg_6,pool_stride=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fluid.nets.img_conv_group(input=arg_1,conv_padding=arg_2,conv_num_filter=arg_3,conv_filter_size=arg_4,conv_act=arg_5,pool_size=arg_6,pool_stride=arg_7,)
results["time_high"] = time.time() - start

print(results)

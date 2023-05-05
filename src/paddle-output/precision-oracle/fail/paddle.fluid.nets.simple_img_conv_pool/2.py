results = dict()
import paddle
import time
float_tensor = paddle.rand([57, 1, 28, 1024], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = 20
arg_4 = 2
arg_5 = 2
arg_6 = "relu"
start = time.time()
results["time_low"] = paddle.fluid.nets.simple_img_conv_pool(input=arg_1,filter_size=arg_2,num_filters=arg_3,pool_size=arg_4,pool_stride=arg_5,act=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.nets.simple_img_conv_pool(input=arg_1,filter_size=arg_2,num_filters=arg_3,pool_size=arg_4,pool_stride=arg_5,act=arg_6,)
results["time_high"] = time.time() - start

print(results)

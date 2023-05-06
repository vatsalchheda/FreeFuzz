results = dict()
import paddle
import time
arg_1 = 32
arg_2 = 1
arg_3 = 3
arg_class = paddle.fluid.dygraph.nn.Conv2DTranspose(num_channels=arg_1,num_filters=arg_2,filter_size=arg_3,)
float_tensor = paddle.rand([3, 32, 32, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_0_tensor = f16_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

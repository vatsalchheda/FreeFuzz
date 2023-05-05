results = dict()
import paddle
import time
arg_1 = True
arg_2 = 52
arg_3 = 3
arg_4 = False
arg_class = paddle.fluid.dygraph.nn.Conv3DTranspose(num_channels=arg_1,num_filters=arg_2,filter_size=arg_3,use_cudnn=arg_4,)
float_tensor = paddle.rand([5, 3, 12, 32, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_0_tensor = f16_tensor
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

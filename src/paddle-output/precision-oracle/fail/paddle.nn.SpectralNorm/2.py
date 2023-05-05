results = dict()
import paddle
import time
arg_1_0 = -52
arg_1_1 = 32
arg_1_2 = 0
arg_1_3 = -1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = True
arg_3 = 2
arg_class = paddle.nn.SpectralNorm(arg_1,dim=arg_2,power_iters=arg_3,)
float_tensor = paddle.rand([2, 8, 32, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_0_tensor = f16_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

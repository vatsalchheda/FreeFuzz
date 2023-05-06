results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0_0 = 0
arg_2_0_1 = 1
arg_2_0_2 = 2
arg_2_0 = [arg_2_0_0,arg_2_0_1,arg_2_0_2,]
arg_2_1_0 = 1
arg_2_1_1 = 2
arg_2_1_2 = 0
arg_2_1 = [arg_2_1_0,arg_2_1_1,arg_2_1_2,]
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = -11
start = time.time()
results["time_low"] = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2_0 = [arg_2_0_0,arg_2_0_1,arg_2_0_2,]
arg_2_1 = [arg_2_1_0,arg_2_1_1,arg_2_1_2,]
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)

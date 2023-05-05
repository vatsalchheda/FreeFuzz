results = dict()
import paddle
import time
arg_1 = 0.0
arg_2 = 0.02
arg_3 = 17
arg_class = paddle.fluid.initializer.NormalInitializer(loc=arg_1,scale=arg_2,seed=arg_3,)
float_tensor = paddle.rand([1024, 1024, 3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_0_tensor = f16_tensor
arg_4_0 = arg_4_0_tensor.clone()
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_1_tensor = f16_tensor
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

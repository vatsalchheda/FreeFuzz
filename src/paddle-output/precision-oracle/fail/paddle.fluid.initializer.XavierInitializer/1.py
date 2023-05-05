results = dict()
import paddle
import time
arg_1 = False
arg_2 = None
arg_3 = None
arg_4 = 0
arg_class = paddle.fluid.initializer.XavierInitializer(uniform=arg_1,fan_in=arg_2,fan_out=arg_3,seed=arg_4,)
float_tensor = paddle.rand([128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_0_tensor = f16_tensor
arg_5_0 = arg_5_0_tensor.clone()
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_1_tensor = f16_tensor
arg_5_1 = arg_5_1_tensor.clone()
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().type(paddle.float32)
arg_5_1 = arg_5_1_tensor.clone().type(paddle.float32)
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

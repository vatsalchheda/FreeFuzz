results = dict()
import paddle
import time
arg_1 = -0.35355339059327373
arg_2 = 0.35355339059327373
arg_3 = 0
arg_4 = -39
arg_5 = 0
arg_6 = -1.0
arg_class = paddle.fluid.initializer.UniformInitializer(low=arg_1,high=arg_2,seed=arg_3,diag_num=arg_4,diag_step=arg_5,diag_val=arg_6,)
float_tensor = paddle.rand([32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_7_0_tensor = f16_tensor
arg_7_0 = arg_7_0_tensor.clone()
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_7_1_tensor = f16_tensor
arg_7_1 = arg_7_1_tensor.clone()
arg_7 = [arg_7_0,arg_7_1,]
start = time.time()
results["time_low"] = arg_class(*arg_7)
results["time_low"] = time.time() - start
arg_7_0 = arg_7_0_tensor.clone().type(paddle.float32)
arg_7_1 = arg_7_1_tensor.clone().type(paddle.float32)
arg_7 = [arg_7_0,arg_7_1,]
start = time.time()
results["time_high"] = arg_class(*arg_7)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1 = 0.0001
arg_class = paddle.regularizer.L1Decay(arg_1,)
float_tensor = paddle.rand([10], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_0_tensor = f16_tensor
arg_2_0 = arg_2_0_tensor.clone()
float_tensor = paddle.rand([10], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_1_tensor = f16_tensor
arg_2_1 = arg_2_1_tensor.clone()
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_2_tensor = f16_tensor
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float32)
arg_2_2 = arg_2_2_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

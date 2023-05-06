results = dict()
import paddle
import time
arg_1 = 16
arg_2 = -16
arg_class = paddle.nn.SimpleRNNCell(arg_1,arg_2,)
float_tensor = paddle.rand([4, 16], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_0_tensor = f16_tensor
arg_3_0 = arg_3_0_tensor.clone()
float_tensor = paddle.rand([4, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_1_tensor = f16_tensor
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

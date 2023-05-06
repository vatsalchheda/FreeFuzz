results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 256, 126, 16], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.nn.functional.avg_pool2d(arg_1,kernel_size=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.avg_pool2d(arg_1,kernel_size=arg_2,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 3, 112, 112], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = -43
arg_2_1 = -29
arg_2_2 = 41
arg_2_3 = -62
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_low"] = paddle.nn.functional.zeropad2d(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.nn.functional.zeropad2d(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

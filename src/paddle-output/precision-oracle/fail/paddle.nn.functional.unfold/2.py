results = dict()
import paddle
import time
float_tensor = paddle.rand([100, 3, 224, 224], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 1
arg_4 = 1
arg_5 = 1
start = time.time()
results["time_low"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)

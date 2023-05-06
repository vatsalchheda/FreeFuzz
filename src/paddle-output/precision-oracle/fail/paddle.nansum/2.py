results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1024
arg_2 = [arg_2_0,]
arg_3 = False
arg_4 = None
start = time.time()
results["time_low"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 0.8
arg_3 = 1
arg_4 = True
start = time.time()
results["time_low"] = paddle.nanquantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nanquantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
results["time_high"] = time.time() - start

print(results)

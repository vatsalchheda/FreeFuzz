results = dict()
import paddle
import time
arg_1 = 64
arg_2 = 1
arg_3 = 18
arg_4 = False
arg_class = paddle.nn.Conv1D(arg_1,arg_2,arg_3,bias_attr=arg_4,)
float_tensor = paddle.rand([1, 64, 41100], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_0_tensor = f16_tensor
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().type(paddle.float32)
arg_5 = [arg_5_0,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

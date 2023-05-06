results = dict()
import paddle
import time
arg_1 = -41
arg_2 = 16
arg_3 = 61
arg_4 = 1
arg_5 = 0
arg_class = paddle.nn.Conv2D(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,)
float_tensor = paddle.rand([64, 6, 14, 14], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_6_0_tensor = f16_tensor
arg_6_0 = arg_6_0_tensor.clone()
arg_6 = [arg_6_0,]
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
arg_6_0 = arg_6_0_tensor.clone().type(paddle.float32)
arg_6 = [arg_6_0,]
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)

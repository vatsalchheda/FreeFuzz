results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 32, 413], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([32, 32, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = 0
arg_5_0 = 1
arg_5 = [arg_5_0,]
arg_6_0 = 57
arg_6 = [arg_6_0,]
arg_7 = 1
arg_8 = "NCL"
start = time.time()
results["time_low"] = paddle.nn.functional.conv1d(arg_1,arg_2,bias=arg_3,padding=arg_4,stride=arg_5,dilation=arg_6,groups=arg_7,data_format=arg_8,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_5 = [arg_5_0,]
arg_6 = [arg_6_0,]
start = time.time()
results["time_high"] = paddle.nn.functional.conv1d(arg_1,arg_2,bias=arg_3,padding=arg_4,stride=arg_5,dilation=arg_6,groups=arg_7,data_format=arg_8,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 512, 136], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([512, 256, 10], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([256], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = None
arg_5 = 1
arg_6 = 1024
arg_7_0 = 5
arg_7 = [arg_7_0,]
arg_8_0 = 1
arg_8 = [arg_8_0,]
arg_9 = 35
arg_10 = "NCL"
start = time.time()
results["time_low"] = paddle.nn.functional.conv1d_transpose(arg_1,arg_2,bias=arg_3,output_size=arg_4,output_padding=arg_5,padding=arg_6,stride=arg_7,dilation=arg_8,groups=arg_9,data_format=arg_10,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_7 = [arg_7_0,]
arg_8 = [arg_8_0,]
start = time.time()
results["time_high"] = paddle.nn.functional.conv1d_transpose(arg_1,arg_2,bias=arg_3,output_size=arg_4,output_padding=arg_5,padding=arg_6,stride=arg_7,dilation=arg_8,groups=arg_9,data_format=arg_10,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 4, 41, 8, 8], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([4, 6, 3, 3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([6], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = 0
arg_5 = 0
arg_6_0 = 1
arg_6_1 = 1
arg_6_2 = 1
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7_0 = 1
arg_7_1 = 1
arg_7_2 = 1
arg_7 = [arg_7_0,arg_7_1,arg_7_2,]
arg_8 = 1
arg_9 = None
arg_10 = "NCDHW"
start = time.time()
results["time_low"] = paddle.nn.functional.conv3d_transpose(arg_1,arg_2,bias=arg_3,padding=arg_4,output_padding=arg_5,stride=arg_6,dilation=arg_7,groups=arg_8,output_size=arg_9,data_format=arg_10,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7 = [arg_7_0,arg_7_1,arg_7_2,]
start = time.time()
results["time_high"] = paddle.nn.functional.conv3d_transpose(arg_1,arg_2,bias=arg_3,padding=arg_4,output_padding=arg_5,stride=arg_6,dilation=arg_7,groups=arg_8,output_size=arg_9,data_format=arg_10,)
results["time_high"] = time.time() - start

print(results)

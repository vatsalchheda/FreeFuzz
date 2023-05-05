results = dict()
import paddle
import time
arg_1 = 5
arg_2 = 4
arg_3 = 1000
arg_class = paddle.nn.Bilinear(in1_features=arg_1,in2_features=arg_2,out_features=arg_3,)
float_tensor = paddle.rand([5, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_0_tensor = f16_tensor
arg_4_0 = arg_4_0_tensor.clone()
float_tensor = paddle.rand([5, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_1_tensor = f16_tensor
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

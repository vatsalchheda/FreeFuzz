results = dict()
import paddle
import time
arg_1 = 128
arg_2 = 2
arg_3 = 67
arg_4 = 28
arg_5 = 512
arg_class = paddle.nn.Transformer(arg_1,arg_2,arg_3,arg_4,arg_5,)
float_tensor = paddle.rand([2, 4, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_6_0_tensor = f16_tensor
arg_6_0 = arg_6_0_tensor.clone()
float_tensor = paddle.rand([2, 6, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_6_1_tensor = f16_tensor
arg_6_1 = arg_6_1_tensor.clone()
float_tensor = paddle.rand([2, 2, 4, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_6_2_tensor = f16_tensor
arg_6_2 = arg_6_2_tensor.clone()
float_tensor = paddle.rand([2, 2, 6, 6], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_6_3_tensor = f16_tensor
arg_6_3 = arg_6_3_tensor.clone()
float_tensor = paddle.rand([2, 2, 6, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_6_4_tensor = f16_tensor
arg_6_4 = arg_6_4_tensor.clone()
arg_6 = [arg_6_0,arg_6_1,arg_6_2,arg_6_3,arg_6_4,]
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
arg_6_0 = arg_6_0_tensor.clone().type(paddle.float32)
arg_6_1 = arg_6_1_tensor.clone().type(paddle.float32)
arg_6_2 = arg_6_2_tensor.clone().type(paddle.float32)
arg_6_3 = arg_6_3_tensor.clone().type(paddle.float32)
arg_6_4 = arg_6_4_tensor.clone().type(paddle.float32)
arg_6 = [arg_6_0,arg_6_1,arg_6_2,arg_6_3,arg_6_4,]
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)

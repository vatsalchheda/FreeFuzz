results = dict()
import paddle
import time
float_tensor = paddle.rand([5, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([5, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([1000, 5, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
float_tensor = paddle.rand([1, 1000], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_tensor = f16_tensor
arg_4 = arg_4_tensor.clone()
arg_5 = None
start = time.time()
results["time_low"] = paddle.nn.functional.bilinear(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_4 = arg_4_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.bilinear(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)

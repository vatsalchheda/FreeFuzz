results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 2, 1, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([1, 2, 35, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = False
start = time.time()
results["time_low"] = paddle.matmul(x=arg_1,y=arg_2,transpose_y=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.matmul(x=arg_1,y=arg_2,transpose_y=arg_3,)
results["time_high"] = time.time() - start

print(results)

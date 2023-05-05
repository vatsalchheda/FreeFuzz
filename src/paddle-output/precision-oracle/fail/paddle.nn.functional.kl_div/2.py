results = dict()
import paddle
import time
float_tensor = paddle.rand([5, 20], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([5, 20], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = "none"
start = time.time()
results["time_low"] = paddle.nn.functional.kl_div(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.kl_div(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)

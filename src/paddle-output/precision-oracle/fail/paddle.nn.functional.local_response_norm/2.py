results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 3, 112, 112], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = "replicate"
arg_3 = 44.0
arg_4 = 0.75
arg_5 = 1.0
arg_6 = "sum"
arg_7 = None
start = time.time()
results["time_low"] = paddle.nn.functional.local_response_norm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.local_response_norm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
results["time_high"] = time.time() - start

print(results)

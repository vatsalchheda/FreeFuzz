results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 35.5
arg_3 = False
arg_4 = None
start = time.time()
results["time_low"] = paddle.nn.functional.alpha_dropout(arg_1,p=arg_2,training=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.alpha_dropout(arg_1,p=arg_2,training=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)

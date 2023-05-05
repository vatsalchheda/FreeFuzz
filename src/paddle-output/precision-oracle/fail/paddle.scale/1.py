results = dict()
import paddle
import time
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = -1.0
arg_3 = "Categorical_entropy"
start = time.time()
results["time_low"] = paddle.scale(arg_1,scale=arg_2,name=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.scale(arg_1,scale=arg_2,name=arg_3,)
results["time_high"] = time.time() - start

print(results)

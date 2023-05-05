results = dict()
import paddle
import time
arg_1 = "debug_func"
float_tensor = paddle.rand([1, 200], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = None
start = time.time()
results["time_low"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
results["time_high"] = time.time() - start

print(results)

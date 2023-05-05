results = dict()
import paddle
import time
arg_1 = "tanh"
float_tensor = paddle.rand([1, 200], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([1, 200], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = "tanh_grad"
float_tensor = paddle.rand([1, 200], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_tensor = f16_tensor
arg_5 = arg_5_tensor.clone()
start = time.time()
results["time_low"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,backward_func=arg_4,skip_vars_in_backward_input=arg_5,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_5 = arg_5_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,backward_func=arg_4,skip_vars_in_backward_input=arg_5,)
results["time_high"] = time.time() - start

print(results)

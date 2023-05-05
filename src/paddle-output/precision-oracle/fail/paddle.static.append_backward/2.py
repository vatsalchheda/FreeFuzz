results = dict()
import paddle
import time
float_tensor = paddle.rand([1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([100, 256, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_0_tensor = f16_tensor
arg_2_0 = arg_2_0_tensor.clone()
float_tensor = paddle.rand([3350, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_1_tensor = f16_tensor
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "builtinsset"
start = time.time()
results["time_low"] = paddle.static.append_backward(loss=arg_1,parameter_list=arg_2,no_grad_set=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.static.append_backward(loss=arg_1,parameter_list=arg_2,no_grad_set=arg_3,)
results["time_high"] = time.time() - start

print(results)

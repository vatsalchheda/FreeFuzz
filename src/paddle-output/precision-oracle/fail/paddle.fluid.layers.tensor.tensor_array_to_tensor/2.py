results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 18
arg_3 = True
start = time.time()
results["time_low"] = paddle.fluid.layers.tensor.tensor_array_to_tensor(arg_1,axis=arg_2,use_stack=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.tensor.tensor_array_to_tensor(arg_1,axis=arg_2,use_stack=arg_3,)
results["time_high"] = time.time() - start

print(results)

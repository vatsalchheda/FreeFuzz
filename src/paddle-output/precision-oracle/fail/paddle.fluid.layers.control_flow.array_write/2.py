results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 10], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([-1, 10], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.fluid.layers.control_flow.array_write(x=arg_1,i=arg_2,array=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.control_flow.array_write(x=arg_1,i=arg_2,array=arg_3,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 6, 13, 13], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = 1
float_tensor = paddle.rand([3, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_tensor = f16_tensor
arg_4 = arg_4_tensor.clone()
start = time.time()
results["time_low"] = paddle.index_add_(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
arg_4 = arg_4_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.index_add_(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)

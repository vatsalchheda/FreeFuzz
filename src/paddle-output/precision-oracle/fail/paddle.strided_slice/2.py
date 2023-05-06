results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 4, 0, 6], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2_2 = 3
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_0_tensor = int8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
arg_4_0 = -60
arg_4_1 = 25
arg_4_2 = 63
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5_0 = 19
arg_5_1 = -1024
arg_5_2 = -16
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
start = time.time()
results["time_low"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0 = arg_3_0_tensor.clone().type(paddle.int32)
arg_3 = [arg_3_0,]
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
start = time.time()
results["time_high"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
results["time_high"] = time.time() - start

print(results)

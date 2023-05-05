results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_class = paddle.text.ViterbiDecoder(arg_1,include_bos_eos_tag=arg_2,)
float_tensor = paddle.rand([2, 4, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_0_tensor = f16_tensor
arg_3_0 = arg_3_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=128, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_1_tensor = int8_tensor
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.int64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

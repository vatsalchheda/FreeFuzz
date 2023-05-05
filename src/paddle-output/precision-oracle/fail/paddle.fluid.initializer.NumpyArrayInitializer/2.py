results = dict()
import paddle
import time
arg_1 = "numpyndarray"
arg_class = paddle.fluid.initializer.NumpyArrayInitializer(arg_1,)
int_tensor = paddle.randint(low=-128, high=128, shape=[6, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_0_tensor = int8_tensor
arg_2_0 = arg_2_0_tensor.clone()
float_tensor = paddle.rand([2, 2], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_1_tensor = f16_tensor
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.int32)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

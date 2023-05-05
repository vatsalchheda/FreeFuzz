results = dict()
import paddle
import time
arg_1 = 1
arg_class = paddle.fluid.dygraph.nn.GRUUnit(size=arg_1,)
int_tensor = paddle.randint(low=0, high=255, shape=[9, 15], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_2_0_tensor = uint8_tensor
arg_2_0 = arg_2_0_tensor.clone()
float_tensor = paddle.rand([9, 0, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_1_tensor = f16_tensor
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.uint8)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

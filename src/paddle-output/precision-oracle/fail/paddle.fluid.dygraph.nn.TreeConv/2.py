results = dict()
import paddle
import time
arg_1 = 5
arg_2 = 6
arg_3 = 1
arg_4 = 36
arg_class = paddle.fluid.dygraph.nn.TreeConv(feature_size=arg_1,output_size=arg_2,num_filters=arg_3,max_depth=arg_4,)
float_tensor = paddle.rand([36, 10, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_0_tensor = f16_tensor
arg_5_0 = arg_5_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=128, shape=[1, 9, 2, 0], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_5_1_tensor = int8_tensor
arg_5_1 = arg_5_1_tensor.clone()
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().type(paddle.float32)
arg_5_1 = arg_5_1_tensor.clone().type(paddle.int32)
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

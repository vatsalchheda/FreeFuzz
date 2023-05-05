results = dict()
import paddle
import time
arg_1 = 5
arg_2 = 6
arg_3 = 1
arg_4 = 2
arg_class = paddle.fluid.dygraph.nn.TreeConv(feature_size=arg_1,output_size=arg_2,num_filters=arg_3,max_depth=arg_4,)
arg_5_0_tensor = paddle.rand([1, 10, 5], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[1, 9, 2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_5_1_tensor = int8_tensor
arg_5_1 = arg_5_1_tensor.clone()
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().astype(paddle.float32)
arg_5_1 = arg_5_1_tensor.clone().astype(paddle.int32)
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

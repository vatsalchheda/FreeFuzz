results = dict()
import paddle
import time
arg_1 = True
arg_2 = "max"
arg_3 = 1
arg_4 = False
arg_class = paddle.fluid.dygraph.nn.Pool2D(pool_size=arg_1,pool_type=arg_2,pool_stride=arg_3,global_pooling=arg_4,)
float_tensor = paddle.rand([3, 32, 32, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_0_tensor = f16_tensor
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().type(paddle.float32)
arg_5 = [arg_5_0,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

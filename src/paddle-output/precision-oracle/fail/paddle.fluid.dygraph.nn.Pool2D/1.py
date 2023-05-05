results = dict()
import paddle
import time
arg_1 = 2
arg_2 = "max"
arg_3 = 2
arg_4 = False
arg_class = paddle.fluid.dygraph.nn.Pool2D(pool_size=arg_1,pool_type=arg_2,pool_stride=arg_3,global_pooling=arg_4,)
arg_5_0_tensor = paddle.randint(-256,2048,[3, 0, 32, 0], dtype=paddle.bfloat16)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().type(paddle.bfloat16)
arg_5 = [arg_5_0,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)

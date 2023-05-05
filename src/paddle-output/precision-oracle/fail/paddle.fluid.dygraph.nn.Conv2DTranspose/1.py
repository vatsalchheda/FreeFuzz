results = dict()
import paddle
import time
arg_1 = 32
arg_2 = 2
arg_3 = "max"
arg_class = paddle.fluid.dygraph.nn.Conv2DTranspose(num_channels=arg_1,num_filters=arg_2,filter_size=arg_3,)
arg_4_0_tensor = paddle.rand([3, 32, 32, 5], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.float32)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

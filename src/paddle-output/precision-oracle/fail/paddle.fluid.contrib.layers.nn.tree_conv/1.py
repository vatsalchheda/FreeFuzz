results = dict()
import paddle
import time
real = paddle.rand([-1, 10, 1024], paddle.float32)
imag = paddle.rand([-1, 10, 1024], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 10, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
arg_4 = -1024
arg_5 = -30
start = time.time()
results["time_low"] = paddle.fluid.contrib.layers.nn.tree_conv(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.complex128)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.contrib.layers.nn.tree_conv(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)

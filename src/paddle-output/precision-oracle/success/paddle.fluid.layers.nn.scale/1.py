results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([16, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = "translated_layer/scale_0"
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.scale(arg_1,arg_2,name=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.scale(arg_1,arg_2,name=arg_3,)
results["time_high"] = time.time() - start

print(results)

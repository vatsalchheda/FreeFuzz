results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,32,[-1, 3, 6, 8, 10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 45
arg_2_1 = 71
arg_2_2 = -49
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = None
arg_4 = None
arg_5 = "TRILINEAR"
arg_6 = None
arg_7 = -38.0
arg_8 = 1
arg_9 = "NCDHW"
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.image_resize(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.image_resize(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,)
results["time_high"] = time.time() - start

print(results)

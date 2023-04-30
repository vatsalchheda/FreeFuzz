results = dict()
import paddle
import time
arg_1 = 0.0
arg_2 = 0.4714045207910317
arg_3 = 20
arg_class = paddle.fluid.initializer.NormalInitializer(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.randint(-8,16,[3, 1, 3, 3], dtype=paddle.float16)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.randint(-2048,32,[2, 2], dtype=paddle.float16)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

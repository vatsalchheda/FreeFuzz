results = dict()
import paddle
import time
arg_1 = 0.9
arg_class = paddle.fluid.initializer.ConstantInitializer(value=arg_1,)
arg_2_0_tensor = paddle.randint(-16384,2048,[1], dtype=paddle.float16)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-4096,8,[2, 2], dtype=paddle.float16)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1 = -0.17677669529663687
arg_2 = 0.17677669529663687
arg_class = paddle.nn.initializer.Uniform(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-64,512,[128, 32], dtype=paddle.float16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-512,1024,[2, 2], dtype=paddle.float16)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float64)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

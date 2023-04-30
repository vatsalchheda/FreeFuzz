results = dict()
import paddle
import time
arg_1 = -30
arg_2 = 0.17677669529663687
arg_3 = 0
arg_4 = 55
arg_5 = 0
arg_6 = 1.0
arg_class = paddle.fluid.initializer.UniformInitializer(low=arg_1,high=arg_2,seed=arg_3,diag_num=arg_4,diag_step=arg_5,diag_val=arg_6,)
arg_7_0_tensor = paddle.randint(-32768,512,[128], dtype=paddle.float16)
arg_7_0 = arg_7_0_tensor.clone()
arg_7_1_tensor = paddle.randint(-4,256,[2, 2], dtype=paddle.float16)
arg_7_1 = arg_7_1_tensor.clone()
arg_7 = [arg_7_0,arg_7_1,]
start = time.time()
results["time_low"] = arg_class(*arg_7)
results["time_low"] = time.time() - start
arg_7_0 = arg_7_0_tensor.clone().type(paddle.float64)
arg_7_1 = arg_7_1_tensor.clone().type(paddle.float32)
arg_7 = [arg_7_0,arg_7_1,]
start = time.time()
results["time_high"] = arg_class(*arg_7)
results["time_high"] = time.time() - start

print(results)

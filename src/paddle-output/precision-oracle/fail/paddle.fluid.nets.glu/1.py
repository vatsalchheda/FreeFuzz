results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,8,[-1, 6, 3, 9], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = "sum"
start = time.time()
results["time_low"] = paddle.fluid.nets.glu(input=arg_1,dim=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.nets.glu(input=arg_1,dim=arg_2,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,32,[4, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 4
start = time.time()
results["time_low"] = paddle.fluid.contrib.sparsity.utils.get_mask_1d(arg_1,n=arg_2,m=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.fluid.contrib.sparsity.utils.get_mask_1d(arg_1,n=arg_2,m=arg_3,)
results["time_high"] = time.time() - start

print(results)

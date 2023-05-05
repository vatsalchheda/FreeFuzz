results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([4, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
arg_3 = 4
start = time.time()
results["time_low"] = paddle.fluid.contrib.sparsity.utils.check_mask_2d(arg_1,n=arg_2,m=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.fluid.contrib.sparsity.utils.check_mask_2d(arg_1,n=arg_2,m=arg_3,)
results["time_high"] = time.time() - start

print(results)

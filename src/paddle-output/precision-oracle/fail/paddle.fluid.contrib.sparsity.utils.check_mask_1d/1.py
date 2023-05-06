results = dict()
import paddle
import time
float_tensor = paddle.rand([4, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = 35
start = time.time()
results["time_low"] = paddle.fluid.contrib.sparsity.utils.check_mask_1d(arg_1,n=arg_2,m=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.fluid.contrib.sparsity.utils.check_mask_1d(arg_1,n=arg_2,m=arg_3,)
results["time_high"] = time.time() - start

print(results)

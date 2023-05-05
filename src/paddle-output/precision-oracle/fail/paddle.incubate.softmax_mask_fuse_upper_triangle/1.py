results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 1, 32, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.incubate.softmax_mask_fuse_upper_triangle(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.incubate.softmax_mask_fuse_upper_triangle(arg_1,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 8, 8, 32], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[3, 4])
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.incubate.softmax_mask_fuse(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.bool)
start = time.time()
results["time_high"] = paddle.incubate.softmax_mask_fuse(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

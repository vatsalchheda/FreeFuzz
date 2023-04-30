results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,32768,[2, 8, 8, 32], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,512,[2, 1, 8, 32], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.incubate.softmax_mask_fuse(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.incubate.softmax_mask_fuse(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

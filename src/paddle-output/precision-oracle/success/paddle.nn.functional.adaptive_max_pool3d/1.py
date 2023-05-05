results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3, 8, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = False
arg_4 = None
start = time.time()
results["time_low"] = paddle.nn.functional.adaptive_max_pool3d(arg_1,output_size=arg_2,return_mask=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.adaptive_max_pool3d(arg_1,output_size=arg_2,return_mask=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)

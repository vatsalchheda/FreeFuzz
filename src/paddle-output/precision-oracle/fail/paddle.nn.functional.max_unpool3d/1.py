results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,64,[3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,1,[1, 1, 2, 2, 3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 20
arg_4 = 0
start = time.time()
results["time_low"] = paddle.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,padding=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,padding=arg_4,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1 = 49
arg_2 = 0
arg_class = paddle.nn.MaxUnPool3D(kernel_size=arg_1,padding=arg_2,)
arg_3_0_tensor = paddle.randint(-32,4096,[1, 1, 2, 2, 3], dtype=paddle.float16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-4,4,[1, 1, 2, 2, 3], dtype=paddle.int8)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.int32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,64,[2, 12, 12], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 155.0
start = time.time()
results["time_low"] = paddle.nn.functional.fold(arg_1,output_sizes=arg_2,kernel_sizes=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.fold(arg_1,output_sizes=arg_2,kernel_sizes=arg_3,)
results["time_high"] = time.time() - start

print(results)

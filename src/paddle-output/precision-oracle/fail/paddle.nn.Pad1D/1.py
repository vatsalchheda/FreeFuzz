results = dict()
import paddle
import time
arg_1 = 2
arg_2 = 110.0
arg_class = paddle.nn.Pad1D(arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.randint(-2048,4096,[1, 51, 190, 1], dtype=paddle.bfloat16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.bfloat16)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1 = 0.5
arg_class = paddle.nn.Dropout2D(p=arg_1,)
arg_2_0_tensor = paddle.randint(-2048,64,[2, 2, 1, 3], dtype=paddle.float16)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

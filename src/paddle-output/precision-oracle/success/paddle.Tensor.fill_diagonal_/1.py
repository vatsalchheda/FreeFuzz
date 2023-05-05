results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[4, 3])
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
start = time.time()
results["time_low"] = paddle.Tensor.fill_diagonal_(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.bool)
start = time.time()
results["time_high"] = paddle.Tensor.fill_diagonal_(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)

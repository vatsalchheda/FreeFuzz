results = dict()
import paddle
import time
arg_class = paddle.nn.MarginRankingLoss()
arg_1_0_tensor = paddle.randint(-1,16384,[2, 2], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-4,4096,[2, 2], dtype=paddle.float16)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.randint(-8192,4096,[2, 2], dtype=paddle.float16)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1_2 = arg_1_2_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)

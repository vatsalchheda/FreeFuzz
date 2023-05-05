results = dict()
import paddle
import time
arg_1 = 4
arg_2 = -49
arg_class = paddle.nn.SimpleRNNCell(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-8192,1,[0, 43], dtype=paddle.bfloat16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([4, 28], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.bfloat16)
arg_3_1 = arg_3_1_tensor.clone().astype(paddle.float64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

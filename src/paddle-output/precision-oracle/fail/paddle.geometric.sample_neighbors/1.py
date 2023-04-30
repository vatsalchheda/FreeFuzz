results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,128,[13], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,64,[11], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-64,32,[4], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4 = 2
start = time.time()
results["time_low"] = paddle.geometric.sample_neighbors(arg_1,arg_2,arg_3,sample_size=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.geometric.sample_neighbors(arg_1,arg_2,arg_3,sample_size=arg_4,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,16,[2, 1], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = -115.0
arg_3 = 1
arg_4 = 0
start = time.time()
results["time_low"] = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
results["time_high"] = time.time() - start

print(results)

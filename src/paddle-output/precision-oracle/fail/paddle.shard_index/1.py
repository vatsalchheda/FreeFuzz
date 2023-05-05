results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 20
arg_3 = 2
arg_4 = -60
start = time.time()
results["time_low"] = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
start = time.time()
results["time_high"] = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
results["time_high"] = time.time() - start

print(results)

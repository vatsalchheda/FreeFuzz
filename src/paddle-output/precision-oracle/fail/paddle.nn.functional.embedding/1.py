results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1026, 8], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = True
arg_5 = None
start = time.time()
results["time_low"] = paddle.nn.functional.embedding(arg_1,weight=arg_2,padding_idx=arg_3,sparse=arg_4,name=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.embedding(arg_1,weight=arg_2,padding_idx=arg_3,sparse=arg_4,name=arg_5,)
results["time_high"] = time.time() - start

print(results)

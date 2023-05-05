results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 5], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 1027.0
arg_3_1 = "mean"
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,shape=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int32)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,shape=arg_3,)
results["time_high"] = time.time() - start

print(results)

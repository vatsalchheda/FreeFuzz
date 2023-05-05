results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_tensor = int8_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = 0.5
arg_5 = "sum"
arg_6 = None
start = time.time()
results["time_low"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,name=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.int8)
arg_3 = arg_3_tensor.clone().astype(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,name=arg_6,)
results["time_high"] = time.time() - start

print(results)

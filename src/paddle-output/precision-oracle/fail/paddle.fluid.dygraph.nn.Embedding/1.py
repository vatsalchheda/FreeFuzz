results = dict()
import paddle
import time
arg_1_0 = 20
arg_1_1 = 32
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "emb.w"
arg_3 = False
arg_class = paddle.fluid.dygraph.nn.Embedding(size=arg_1,param_attr=arg_2,is_sparse=arg_3,)
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_4_0_tensor = int8_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.int64)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

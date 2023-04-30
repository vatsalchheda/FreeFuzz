results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,64,[3, 1], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,8,[10, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = False
arg_4 = "embedding"
start = time.time()
results["time_low"] = paddle.nn.functional.embedding(x=arg_1,weight=arg_2,sparse=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.embedding(x=arg_1,weight=arg_2,sparse=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)

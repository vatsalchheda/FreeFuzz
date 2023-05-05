results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -47
arg_3 = 31
arg_4 = 49
start = time.time()
results["time_low"] = paddle.nn.functional.diag_embed(arg_1,offset=arg_2,dim1=arg_3,dim2=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.diag_embed(arg_1,offset=arg_2,dim1=arg_3,dim2=arg_4,)
results["time_high"] = time.time() - start

print(results)

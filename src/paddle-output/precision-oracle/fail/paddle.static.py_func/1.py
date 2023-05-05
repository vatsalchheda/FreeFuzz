results = dict()
import paddle
import time
arg_1 = "replicate"
arg_2_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
start = time.time()
results["time_low"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
results["time_high"] = time.time() - start

print(results)

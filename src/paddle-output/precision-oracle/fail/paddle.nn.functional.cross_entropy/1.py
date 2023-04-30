results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,32768,[4, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,16384,[0, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = True
arg_4 = -1
arg_5 = None
arg_6 = "mean"
start = time.time()
results["time_low"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,soft_label=arg_3,axis=arg_4,weight=arg_5,reduction=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,soft_label=arg_3,axis=arg_4,weight=arg_5,reduction=arg_6,)
results["time_high"] = time.time() - start

print(results)

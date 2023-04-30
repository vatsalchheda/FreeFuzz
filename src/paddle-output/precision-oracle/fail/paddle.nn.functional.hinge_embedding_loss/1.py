results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,64,[3, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,8,[3, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
arg_4 = -28.0
arg_5 = None
start = time.time()
results["time_low"] = paddle.nn.functional.hinge_embedding_loss(arg_1,arg_2,reduction=arg_3,margin=arg_4,name=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.hinge_embedding_loss(arg_1,arg_2,reduction=arg_3,margin=arg_4,name=arg_5,)
results["time_high"] = time.time() - start

print(results)

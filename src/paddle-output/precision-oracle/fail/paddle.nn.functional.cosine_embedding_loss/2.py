results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,64,[2, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,1024,[2, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32768,16,[3, 3], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.5
arg_5 = "mean"
arg_6 = None
start = time.time()
results["time_low"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,name=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,name=arg_6,)
results["time_high"] = time.time() - start

print(results)

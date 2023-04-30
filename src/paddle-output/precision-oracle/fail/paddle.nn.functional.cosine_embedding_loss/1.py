results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,128,[2, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,32,[2, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4,32,[2], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.5
arg_5 = "mean"
start = time.time()
results["time_low"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,)
results["time_high"] = time.time() - start

print(results)

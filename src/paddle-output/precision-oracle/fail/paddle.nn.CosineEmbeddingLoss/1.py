results = dict()
import paddle
import time
arg_1 = 0.5
arg_2 = "mean"
arg_class = paddle.nn.CosineEmbeddingLoss(margin=arg_1,reduction=arg_2,)
arg_3_0_tensor = paddle.randint(-1024,256,[2, 3], dtype=paddle.float16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-2048,16,[2, 3], dtype=paddle.float16)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.randint(-16,16,[2], dtype=paddle.int8)
arg_3_2 = arg_3_2_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.float32)
arg_3_2 = arg_3_2_tensor.clone().type(paddle.int64)
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

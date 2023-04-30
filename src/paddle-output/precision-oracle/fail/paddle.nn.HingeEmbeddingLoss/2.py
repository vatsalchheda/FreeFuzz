results = dict()
import paddle
import time
arg_1 = -1.0
arg_2 = "none"
arg_class = paddle.nn.HingeEmbeddingLoss(margin=arg_1,reduction=arg_2,)
arg_3_0_tensor = paddle.randint(-1024,256,[3, 3], dtype=paddle.float16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-128,8,[3, 3], dtype=paddle.float16)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)

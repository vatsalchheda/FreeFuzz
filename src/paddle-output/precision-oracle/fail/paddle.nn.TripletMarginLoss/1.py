results = dict()
import paddle
import time
arg_1 = 1.0
arg_2 = True
arg_3 = -1e+20
arg_class = paddle.nn.TripletMarginLoss(margin=arg_1,swap=arg_2,reduction=arg_3,)
arg_4_0_tensor = paddle.rand([0, 3], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4_2_tensor = paddle.rand([3, 0], dtype=paddle.float32)
arg_4_2 = arg_4_2_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().astype(paddle.float32)
arg_4_2 = arg_4_2_tensor.clone().astype(paddle.float32)
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

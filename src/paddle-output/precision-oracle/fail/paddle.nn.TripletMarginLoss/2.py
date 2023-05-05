results = dict()
import paddle
import time
arg_1 = 1.0
arg_2 = True
arg_3 = "mean"
arg_class = paddle.nn.TripletMarginLoss(margin=arg_1,swap=arg_2,reduction=arg_3,)
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_0_tensor = f16_tensor
arg_4_0 = arg_4_0_tensor.clone()
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_1_tensor = f16_tensor
arg_4_1 = arg_4_1_tensor.clone()
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_2_tensor = f16_tensor
arg_4_2 = arg_4_2_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().type(paddle.float32)
arg_4_2 = arg_4_2_tensor.clone().type(paddle.float32)
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)

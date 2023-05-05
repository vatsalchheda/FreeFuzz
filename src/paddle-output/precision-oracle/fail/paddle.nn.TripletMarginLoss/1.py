results = dict()
import paddle
import time
arg_1 = True
arg_class = paddle.nn.TripletMarginLoss(reduction=arg_1,)
float_tensor = paddle.rand([37, 34], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_0_tensor = f16_tensor
arg_2_0 = arg_2_0_tensor.clone()
float_tensor = paddle.rand([3, 3, 0], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_1_tensor = f16_tensor
arg_2_1 = arg_2_1_tensor.clone()
float_tensor = paddle.rand([0, 0, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_2_tensor = f16_tensor
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float32)
arg_2_2 = arg_2_2_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)

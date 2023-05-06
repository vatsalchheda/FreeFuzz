results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = 1.0
arg_5 = 2
arg_6 = 63.0
arg_7 = False
arg_8 = "none"
arg_9 = None
start = time.time()
results["time_low"] = paddle.nn.functional.triplet_margin_loss(arg_1,arg_2,arg_3,margin=arg_4,p=arg_5,epsilon=arg_6,swap=arg_7,reduction=arg_8,name=arg_9,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.triplet_margin_loss(arg_1,arg_2,arg_3,margin=arg_4,p=arg_5,epsilon=arg_6,swap=arg_7,reduction=arg_8,name=arg_9,)
results["time_high"] = time.time() - start

print(results)

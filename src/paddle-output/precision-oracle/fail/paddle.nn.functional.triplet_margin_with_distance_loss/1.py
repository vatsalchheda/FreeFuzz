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
int_tensor = paddle.randint(low=-128, high=127, shape=[3, 3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_tensor = int8_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = 1.0
arg_5 = False
arg_6 = "mean"
arg_7 = None
start = time.time()
results["time_low"] = paddle.nn.functional.triplet_margin_with_distance_loss(arg_1,arg_2,arg_3,margin=arg_4,swap=arg_5,reduction=arg_6,name=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int16)
start = time.time()
results["time_high"] = paddle.nn.functional.triplet_margin_with_distance_loss(arg_1,arg_2,arg_3,margin=arg_4,swap=arg_5,reduction=arg_6,name=arg_7,)
results["time_high"] = time.time() - start

print(results)

results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 3, 6, 9], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([-1, 3, 9, 12], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3_0 = 52.0
arg_3 = [arg_3_0,]
arg_4 = None
arg_5_0 = 1.0
arg_5 = [arg_5_0,]
arg_6_0 = -3.8999999999999986
arg_6_1 = 0.0
arg_6_2 = 37.2
arg_6_3 = -1e+20
arg_6 = [arg_6_0,arg_6_1,arg_6_2,arg_6_3,]
arg_7 = False
arg_8 = False
arg_9_0 = 63.0
arg_9_1 = 1.0
arg_9 = [arg_9_0,arg_9_1,]
arg_10 = 44.0
arg_11 = False
arg_12_0 = 4
arg_12_1 = 26
arg_12 = (arg_12_0,arg_12_1,)
start = time.time()
results["time_low"] = paddle.vision.ops.prior_box(input=arg_1,image=arg_2,min_sizes=arg_3,max_sizes=arg_4,aspect_ratios=arg_5,variance=arg_6,flip=arg_7,clip=arg_8,steps=arg_9,offset=arg_10,min_max_aspect_ratios_order=arg_11,name=arg_12,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,]
arg_5 = [arg_5_0,]
arg_6 = [arg_6_0,arg_6_1,arg_6_2,arg_6_3,]
arg_9 = [arg_9_0,arg_9_1,]
arg_12 = (arg_12_0,arg_12_1,)
start = time.time()
results["time_high"] = paddle.vision.ops.prior_box(input=arg_1,image=arg_2,min_sizes=arg_3,max_sizes=arg_4,aspect_ratios=arg_5,variance=arg_6,flip=arg_7,clip=arg_8,steps=arg_9,offset=arg_10,min_max_aspect_ratios_order=arg_11,name=arg_12,)
results["time_high"] = time.time() - start

print(results)

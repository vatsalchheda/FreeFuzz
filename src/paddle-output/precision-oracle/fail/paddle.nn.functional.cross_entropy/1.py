results = dict()
import paddle
import time
float_tensor = paddle.rand([64, 10], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[64, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = -158
arg_5 = "sum"
arg_6 = False
arg_7 = 120.0
arg_8 = True
arg_9 = -972.0
start = time.time()
results["time_low"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,weight=arg_3,ignore_index=arg_4,reduction=arg_5,soft_label=arg_6,axis=arg_7,use_softmax=arg_8,name=arg_9,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,weight=arg_3,ignore_index=arg_4,reduction=arg_5,soft_label=arg_6,axis=arg_7,use_softmax=arg_8,name=arg_9,)
results["time_high"] = time.time() - start

print(results)

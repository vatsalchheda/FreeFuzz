results = dict()
import paddle
import time
float_tensor = paddle.rand([3, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
int_tensor = paddle.randint(low=-128, high=128, shape=[4], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_tensor = int8_tensor
arg_3 = arg_3_tensor.clone()
int_tensor = paddle.randint(low=-128, high=128, shape=[4], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_4_tensor = int8_tensor
arg_4 = arg_4_tensor.clone()
arg_5 = True
arg_6 = "sum"
start = time.time()
results["time_low"] = paddle.geometric.send_ue_recv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,reduce_op=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int32)
arg_4 = arg_4_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.geometric.send_ue_recv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,reduce_op=arg_6,)
results["time_high"] = time.time() - start

print(results)

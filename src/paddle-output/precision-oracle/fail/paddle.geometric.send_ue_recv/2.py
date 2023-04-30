results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,512,[3, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,32,[4], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-128,16,[4], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-16,2,[4], dtype=paddle.int8)
arg_4 = arg_4_tensor.clone()
arg_5 = "add"
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

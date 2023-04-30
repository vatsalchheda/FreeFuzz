results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2048,128,[3, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,32,[3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16,8,[3], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4 = "sum"
start = time.time()
results["time_low"] = paddle.geometric.send_u_recv(arg_1,arg_2,arg_3,reduce_op=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
arg_3 = arg_3_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.geometric.send_u_recv(arg_1,arg_2,arg_3,reduce_op=arg_4,)
results["time_high"] = time.time() - start

print(results)

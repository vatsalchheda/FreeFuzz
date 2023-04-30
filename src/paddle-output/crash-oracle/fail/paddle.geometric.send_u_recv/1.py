import paddle
arg_1_tensor = paddle.randint(-1024,128,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,2048,[3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4,16,[3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4 = "sum"
res = paddle.geometric.send_u_recv(arg_1,arg_2,arg_3,reduce_op=arg_4,)

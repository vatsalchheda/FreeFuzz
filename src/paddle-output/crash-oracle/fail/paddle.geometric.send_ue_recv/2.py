import paddle
arg_1_tensor = paddle.randint(-256,32768,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,4096,[4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4,8192,[4], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-8192,1024,[4], dtype=paddle.int32)
arg_4 = arg_4_tensor.clone()
arg_5 = "add"
arg_6 = "sum"
res = paddle.geometric.send_ue_recv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,reduce_op=arg_6,)

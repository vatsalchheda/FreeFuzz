import paddle
arg_1_tensor = paddle.randint(-8192,4,[1, 6, 1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,1,[3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-512,256,[3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-1024,32,[4], dtype=paddle.int32)
arg_4 = arg_4_tensor.clone()
arg_5 = True
arg_6 = "sum"
arg_7_tensor = paddle.randint(-256,16384,[1], dtype=paddle.int32)
arg_7 = arg_7_tensor.clone()
res = paddle.geometric.send_ue_recv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,reduce_op=arg_6,out_size=arg_7,)

import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2048, 1024, [3], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-64, 512, [3], dtype=paddle.int32arg_4 = arg_4_tensor.clone()
arg_5 = -83.0
arg_6 = "sum"
arg_7_tensor = paddle.randint(-32, 8, [1], dtype=paddle.int32arg_7 = arg_7_tensor.clone()
res = paddle.geometric.send_ue_recv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,reduce_op=arg_6,out_size=arg_7,)

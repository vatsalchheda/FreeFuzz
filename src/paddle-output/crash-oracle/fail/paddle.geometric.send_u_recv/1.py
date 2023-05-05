import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192, 16384, [4], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1, 2048, [4], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
arg_4 = "max"
res = paddle.geometric.send_u_recv(arg_1,arg_2,arg_3,reduce_op=arg_4,)

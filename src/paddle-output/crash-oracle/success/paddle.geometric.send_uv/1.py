import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-512,16,[4], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-1024,1024,[4], dtype=paddle.int32)
arg_4 = arg_4_tensor.clone()
arg_5 = "add"
res = paddle.geometric.send_uv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,)

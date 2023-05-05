import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,1,[3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-128,4096,[3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4 = "sum"
arg_5_tensor = paddle.randint(-32,32,[1], dtype=paddle.int32)
arg_5 = arg_5_tensor.clone()
res = paddle.incubate.graph_send_recv(arg_1,arg_2,arg_3,pool_type=arg_4,out_size=arg_5,)

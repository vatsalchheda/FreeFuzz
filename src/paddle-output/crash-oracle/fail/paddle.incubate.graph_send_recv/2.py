import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 2048, [3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32768, 2048, [4], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4 = "sum"
res = paddle.incubate.graph_send_recv(arg_1,arg_2,arg_3,pool_type=arg_4,)

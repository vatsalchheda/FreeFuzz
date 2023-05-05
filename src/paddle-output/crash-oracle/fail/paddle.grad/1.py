import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = True
arg_5 = True
res = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)

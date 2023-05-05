import paddle
arg_1_0_tensor = paddle.rand([], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_0_tensor = paddle.rand([2], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = False
res = paddle.grad(arg_1,arg_2,create_graph=arg_3,)

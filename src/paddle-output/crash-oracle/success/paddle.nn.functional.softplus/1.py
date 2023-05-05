import paddle
arg_1_tensor = paddle.rand([4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = 20
arg_4_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
res = paddle.nn.functional.softplus(arg_1,arg_2,arg_3,arg_4,)

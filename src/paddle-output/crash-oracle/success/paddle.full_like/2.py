import paddle
arg_1_tensor = paddle.rand([10, 8, 22, 22], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
res = paddle.full_like(arg_1,fill_value=arg_2,)

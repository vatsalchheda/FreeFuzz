import paddle
arg_1_tensor = paddle.rand([4, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 1, [2, 1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.Tensor.fill_diagonal_tensor(arg_1,arg_2,)

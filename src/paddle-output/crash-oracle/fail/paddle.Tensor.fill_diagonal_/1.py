import paddle
arg_1_tensor = paddle.randint(-4096, 16, [4, 3, 0], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
res = paddle.Tensor.fill_diagonal_(arg_1,arg_2,)

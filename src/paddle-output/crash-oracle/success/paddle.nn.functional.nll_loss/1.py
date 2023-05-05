import paddle
arg_1_tensor = paddle.rand([16, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1, 4096, [16, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.nll_loss(arg_1,arg_2,)

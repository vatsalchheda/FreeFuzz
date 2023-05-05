import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
res = paddle.count_nonzero(arg_1,axis=arg_2,)

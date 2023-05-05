import paddle
arg_1_tensor = paddle.randint(-1,128,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.Tensor.fill_(arg_1,arg_2,)

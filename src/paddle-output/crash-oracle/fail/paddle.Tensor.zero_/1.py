import paddle
arg_1_tensor = paddle.randint(-16, 2048, [5], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.zero_(arg_1,)

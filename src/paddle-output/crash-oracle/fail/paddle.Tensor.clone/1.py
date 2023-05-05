import paddle
arg_1_tensor = paddle.randint(-4, 2, [1, 29960, 0], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.clone(arg_1,)

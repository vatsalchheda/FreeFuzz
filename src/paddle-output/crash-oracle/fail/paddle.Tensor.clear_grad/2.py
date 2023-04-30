import paddle
arg_1_tensor = paddle.randint(-1,128,[2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.clear_grad(arg_1,)

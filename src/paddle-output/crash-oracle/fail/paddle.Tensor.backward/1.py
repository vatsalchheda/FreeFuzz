import paddle
arg_1_tensor = paddle.randint(-4,8,[1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.backward(arg_1,)

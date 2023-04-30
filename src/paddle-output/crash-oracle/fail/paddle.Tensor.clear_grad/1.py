import paddle
arg_1_tensor = paddle.randint(-32768,128,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.clear_grad(arg_1,)

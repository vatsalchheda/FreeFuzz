import paddle
arg_1_tensor = paddle.randint(-16,8192,[2], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.backward(arg_1,)

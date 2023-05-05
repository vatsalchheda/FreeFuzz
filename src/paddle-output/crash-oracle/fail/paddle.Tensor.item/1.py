import paddle
arg_1_tensor = paddle.randint(-512,8192,[1], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.item(arg_1,)

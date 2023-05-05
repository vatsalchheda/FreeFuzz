import paddle
arg_1_tensor = paddle.randint(-32,1,[6, 8], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.cpu(arg_1,)

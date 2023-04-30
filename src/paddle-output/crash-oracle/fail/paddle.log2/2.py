import paddle
arg_1_tensor = paddle.randint(-4096,1,[1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.log2(arg_1,)

import paddle
arg_1_tensor = paddle.randint(-128,8192,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.dim(arg_1,)

import paddle
arg_1_tensor = paddle.randint(-2048,32768,[3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.dim(arg_1,)

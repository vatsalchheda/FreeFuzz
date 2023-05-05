import paddle
arg_1_tensor = paddle.rand([4, 1, 1, 11], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "float32"
res = paddle.Tensor.astype(arg_1,arg_2,)

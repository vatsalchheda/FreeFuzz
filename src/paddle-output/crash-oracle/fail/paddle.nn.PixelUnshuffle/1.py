import paddle
arg_1 = 7
arg_class = paddle.nn.PixelUnshuffle(arg_1,)
arg_2_0_tensor = paddle.rand([2, 1, 12, 12], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)

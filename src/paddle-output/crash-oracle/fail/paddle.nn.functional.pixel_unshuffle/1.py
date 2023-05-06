import paddle
arg_1_tensor = paddle.rand([64, 120], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 51
res = paddle.nn.functional.pixel_unshuffle(arg_1,arg_2,)

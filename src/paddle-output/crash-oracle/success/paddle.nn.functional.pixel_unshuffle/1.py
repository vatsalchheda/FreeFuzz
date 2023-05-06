import paddle
arg_1_tensor = paddle.rand([21, 1, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = "NCHW"
arg_4 = None
res = paddle.nn.functional.pixel_unshuffle(arg_1,arg_2,arg_3,arg_4,)

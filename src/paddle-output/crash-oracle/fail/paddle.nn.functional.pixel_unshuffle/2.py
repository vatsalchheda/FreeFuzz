import paddle
arg_1_tensor = paddle.randint(-2,8,[4, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 14
res = paddle.nn.functional.pixel_unshuffle(arg_1,arg_2,)

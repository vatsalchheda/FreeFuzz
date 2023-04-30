import paddle
arg_1_tensor = paddle.randint(-256,16384,[2, 1, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
res = paddle.nn.functional.pixel_unshuffle(arg_1,arg_2,)

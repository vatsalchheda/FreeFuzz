import paddle
arg_1_tensor = paddle.rand([3, 9, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1024
arg_3 = -16
res = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)

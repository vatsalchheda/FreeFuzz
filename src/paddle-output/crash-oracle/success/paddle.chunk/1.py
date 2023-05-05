import paddle
arg_1_tensor = paddle.rand([1, 128, 41100], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -1
res = paddle.chunk(arg_1,arg_2,axis=arg_3,)

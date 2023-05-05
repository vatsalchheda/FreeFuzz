import paddle
arg_1_tensor = paddle.rand([1, 257, 295], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2.0
res = paddle.pow(arg_1,arg_2,)

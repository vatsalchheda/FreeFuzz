import paddle
arg_1_tensor = paddle.rand([498, 257], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([257, 80], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.mm(arg_1,arg_2,)

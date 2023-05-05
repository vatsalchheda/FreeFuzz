import paddle
arg_1_tensor = paddle.rand([7], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([16, 164, 64], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.pow(arg_1,arg_2,)

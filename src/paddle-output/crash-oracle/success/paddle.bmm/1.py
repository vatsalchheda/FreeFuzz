import paddle
arg_1_tensor = paddle.rand([16, 164, 164], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([16, 164, 64], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.bmm(arg_1,arg_2,)

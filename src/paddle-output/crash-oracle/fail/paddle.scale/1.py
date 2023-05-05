import paddle
arg_1_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 63.0
arg_3_tensor = paddle.rand([192], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.scale(arg_1,scale=arg_2,bias=arg_3,)

import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 1.0
arg_5 = 77.0
res = paddle.nn.functional.triplet_margin_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,)

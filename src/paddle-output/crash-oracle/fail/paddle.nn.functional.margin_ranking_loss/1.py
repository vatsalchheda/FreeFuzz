import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = -32.0
arg_5 = "mean"
arg_6 = None
res = paddle.nn.functional.margin_ranking_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)

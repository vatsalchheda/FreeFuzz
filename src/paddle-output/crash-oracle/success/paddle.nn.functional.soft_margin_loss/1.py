import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
arg_4 = None
res = paddle.nn.functional.soft_margin_loss(arg_1,arg_2,arg_3,arg_4,)

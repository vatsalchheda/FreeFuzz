import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
arg_4 = -8.0
arg_5 = None
res = paddle.nn.functional.smooth_l1_loss(arg_1,arg_2,reduction=arg_3,delta=arg_4,name=arg_5,)

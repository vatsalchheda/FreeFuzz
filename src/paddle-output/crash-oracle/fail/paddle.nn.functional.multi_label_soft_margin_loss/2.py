import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = False
arg_5 = None
res = paddle.nn.functional.multi_label_soft_margin_loss(arg_1,arg_2,weight=arg_3,reduction=arg_4,name=arg_5,)

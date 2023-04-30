import paddle
arg_1_tensor = paddle.randint(-2,2,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,2048,[3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
res = paddle.nn.functional.multi_label_soft_margin_loss(arg_1,arg_2,reduction=arg_3,)

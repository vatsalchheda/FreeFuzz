import paddle
arg_1_tensor = paddle.rand([-1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
res = paddle.incubate.identity_loss(arg_1,reduction=arg_2,)

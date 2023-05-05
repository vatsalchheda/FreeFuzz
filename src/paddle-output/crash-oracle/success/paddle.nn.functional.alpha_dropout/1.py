import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = False
arg_4 = None
res = paddle.nn.functional.alpha_dropout(arg_1,p=arg_2,training=arg_3,name=arg_4,)

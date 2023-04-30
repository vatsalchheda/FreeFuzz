import paddle
arg_1_tensor = paddle.randint(-8192,8192,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,2048,[2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = "none"
arg_4 = 1.0
arg_5 = None
res = paddle.nn.functional.hinge_embedding_loss(arg_1,arg_2,reduction=arg_3,margin=arg_4,name=arg_5,)

import paddle
arg_1_tensor = paddle.randint(-4096, 128, [1, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "sum"
arg_4 = None
res = paddle.nn.functional.l1_loss(arg_1,arg_2,arg_3,name=arg_4,)

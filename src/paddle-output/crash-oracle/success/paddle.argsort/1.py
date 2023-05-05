import paddle
arg_1_tensor = paddle.rand([64, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
res = paddle.argsort(arg_1,descending=arg_2,)

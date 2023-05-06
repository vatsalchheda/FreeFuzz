import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1.0
arg_3 = 1.0
res = paddle.clip(x=arg_1,min=arg_2,max=arg_3,)

import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 21
arg_3 = None
res = paddle.roll(arg_1,arg_2,name=arg_3,)

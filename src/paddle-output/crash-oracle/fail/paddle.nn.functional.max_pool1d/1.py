import paddle
arg_1_tensor = paddle.rand([1, 3, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 53
arg_4 = -31
arg_5 = True
arg_6 = False
arg_7 = None
res = paddle.nn.functional.max_pool1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)

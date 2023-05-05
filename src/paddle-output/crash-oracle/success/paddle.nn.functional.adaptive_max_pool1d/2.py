import paddle
arg_1_tensor = paddle.rand([1, 3, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 60
arg_3 = False
arg_4 = None
res = paddle.nn.functional.adaptive_max_pool1d(arg_1,arg_2,arg_3,arg_4,)

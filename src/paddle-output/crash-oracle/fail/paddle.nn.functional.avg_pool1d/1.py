import paddle
arg_1_tensor = paddle.rand([1, 3, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 2
arg_4 = 0
arg_5 = False
arg_6 = "max"
arg_7 = None
res = paddle.nn.functional.avg_pool1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)

import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2, 512, [2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 79
arg_4 = -36.0
res = paddle.index_add(arg_1,arg_2,arg_3,arg_4,)

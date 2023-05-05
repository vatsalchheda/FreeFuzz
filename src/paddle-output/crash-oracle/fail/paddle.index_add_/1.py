import paddle
arg_1_tensor = paddle.rand([-1, 6, 13, 13], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 512, [2], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
res = paddle.index_add_(arg_1,arg_2,arg_3,arg_4,)

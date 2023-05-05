import paddle
arg_1_tensor = paddle.rand([0], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 3.0
arg_5 = True
res = paddle.sparse.addmm(arg_1,arg_2,arg_3,arg_4,arg_5,)

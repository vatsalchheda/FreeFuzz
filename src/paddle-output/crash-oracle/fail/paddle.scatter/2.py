import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1, 4, [4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 9, 10], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "max"
res = paddle.scatter(arg_1,arg_2,arg_3,overwrite=arg_4,)

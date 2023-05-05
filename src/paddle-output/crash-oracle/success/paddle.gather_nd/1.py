import paddle
arg_1_tensor = paddle.rand([45, 40000], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256, 2, [1, 1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.gather_nd(arg_1,arg_2,)

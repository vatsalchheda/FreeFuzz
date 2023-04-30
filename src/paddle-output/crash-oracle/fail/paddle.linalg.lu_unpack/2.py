import paddle
arg_1_tensor = paddle.randint(-8192,8,[3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,4,[2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.linalg.lu_unpack(arg_1,arg_2,)

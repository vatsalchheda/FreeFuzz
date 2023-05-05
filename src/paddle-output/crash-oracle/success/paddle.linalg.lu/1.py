import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.linalg.lu(arg_1,get_infos=arg_2,)

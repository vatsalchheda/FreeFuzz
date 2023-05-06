import paddle
arg_1_tensor = paddle.rand([3, 4, 5, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.01
arg_3 = True
res = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)

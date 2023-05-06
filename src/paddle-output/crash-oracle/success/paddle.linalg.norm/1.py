import paddle
arg_1_tensor = paddle.rand([1, 3, 32, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -1
res = paddle.linalg.norm(arg_1,p=arg_2,axis=arg_3,)

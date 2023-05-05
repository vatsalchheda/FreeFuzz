import paddle
arg_1_tensor = paddle.rand([3, 4, 5, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512, 32, [3, 3], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.kron(arg_1,arg_2,)

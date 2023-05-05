import paddle
arg_1_tensor = paddle.randint(-1024, 8, [1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = -5
arg_3 = 5
res = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)

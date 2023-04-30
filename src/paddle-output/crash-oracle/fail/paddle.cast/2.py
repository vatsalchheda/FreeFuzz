import paddle
arg_1_tensor = paddle.randint(0,2,[3, 3], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2 = "int64"
res = paddle.cast(arg_1,arg_2,)

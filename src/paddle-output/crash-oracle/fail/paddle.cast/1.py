import paddle
arg_1_tensor = paddle.randint(-128,1,[0, 3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = "int64"
res = paddle.cast(arg_1,arg_2,)

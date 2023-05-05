import paddle
arg_1_tensor = paddle.randint(-2048, 8, [3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "int32"
arg_3 = "float64"
res = paddle.sparse.cast(arg_1,arg_2,arg_3,)

import paddle
arg_1_tensor = paddle.randint(-32768,2,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "sum"
arg_3 = 67
res = paddle.sparse.cast(arg_1,arg_2,arg_3,)

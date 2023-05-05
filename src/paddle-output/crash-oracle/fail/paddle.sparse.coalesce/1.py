import paddle
arg_1_tensor = paddle.randint(-8192,32,[2, 3], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
res = paddle.sparse.coalesce(arg_1,)

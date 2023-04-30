import paddle
arg_1_tensor = paddle.randint(-1024,2,[2, 1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = "float64"
res = paddle.Tensor.astype(arg_1,arg_2,)

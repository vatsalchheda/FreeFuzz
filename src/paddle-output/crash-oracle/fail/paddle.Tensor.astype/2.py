import paddle
arg_1_tensor = paddle.randint(-512, 8192, [4, 1, 1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = "replicate"
res = paddle.Tensor.astype(arg_1,arg_2,)

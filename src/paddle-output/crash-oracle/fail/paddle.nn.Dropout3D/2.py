import paddle
arg_1 = "max"
arg_class = paddle.nn.Dropout3D(p=arg_1,)
arg_2_0_tensor = paddle.rand([1, 4, 2, 1, 3], dtype=paddle.float64)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)

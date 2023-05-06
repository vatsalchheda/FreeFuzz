import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = "Categorical_entropy"
res = paddle.scale(arg_1,scale=arg_2,name=arg_3,)

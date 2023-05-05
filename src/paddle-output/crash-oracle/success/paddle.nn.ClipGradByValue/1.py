import paddle
arg_1 = -1
arg_2 = 1
arg_class = paddle.nn.ClipGradByValue(min=arg_1,max=arg_2,)
arg_3_0_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)

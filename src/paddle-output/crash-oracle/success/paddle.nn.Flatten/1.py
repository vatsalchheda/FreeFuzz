import paddle
arg_1 = 1
arg_class = paddle.nn.Flatten(arg_1,)
arg_2_0_tensor = paddle.rand([32, 1, 28, 28], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)

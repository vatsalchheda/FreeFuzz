import paddle
arg_1_tensor = paddle.rand([1, 512, 2048], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -50
arg_2_1 = -56
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.nn.functional.swish(arg_1,arg_2,)

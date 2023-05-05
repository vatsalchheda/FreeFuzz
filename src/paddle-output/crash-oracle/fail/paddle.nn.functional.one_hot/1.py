import paddle
arg_1_tensor = paddle.rand([4, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -33
res = paddle.nn.functional.one_hot(arg_1,arg_2,)

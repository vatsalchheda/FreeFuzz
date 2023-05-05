import paddle
arg_1_tensor = paddle.rand([1, 255, 274, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -40
res = paddle.pow(arg_1,arg_2,)

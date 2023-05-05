import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.67
arg_3 = 1.72
res = paddle.stanh(arg_1,scale_a=arg_2,scale_b=arg_3,)

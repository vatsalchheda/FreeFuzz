import paddle
arg_1 = 8
arg_2_tensor = paddle.rand([0], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.autograd.jvp(arg_1,arg_2,)

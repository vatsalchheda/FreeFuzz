import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 56.2
arg_3 = None
res = paddle.nn.functional.elu(arg_1,arg_2,arg_3,)

import paddle
arg_1_tensor = paddle.rand([4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = None
res = paddle.nn.functional.softshrink(arg_1,arg_2,arg_3,)

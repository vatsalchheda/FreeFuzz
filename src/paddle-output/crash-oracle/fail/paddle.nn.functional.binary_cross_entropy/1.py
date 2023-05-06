import paddle
arg_1_tensor = paddle.rand([16, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = "mean"
arg_5 = None
res = paddle.nn.functional.binary_cross_entropy(arg_1,arg_2,arg_3,arg_4,arg_5,)

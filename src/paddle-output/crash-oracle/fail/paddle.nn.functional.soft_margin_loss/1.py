import paddle
arg_1_tensor = paddle.rand([5, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128, 2048, [5, 5], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3 = 56.0
arg_4 = None
res = paddle.nn.functional.soft_margin_loss(arg_1,arg_2,arg_3,arg_4,)

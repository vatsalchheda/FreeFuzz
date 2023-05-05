import paddle
arg_1_tensor = paddle.rand([5, 20], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([5, 20], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "sum"
res = paddle.nn.functional.kl_div(arg_1,arg_2,arg_3,)

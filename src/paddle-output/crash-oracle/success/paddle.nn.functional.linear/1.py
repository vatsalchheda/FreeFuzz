import paddle
arg_1_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([128, 128], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = None
res = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,name=arg_4,)

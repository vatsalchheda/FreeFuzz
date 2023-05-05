import paddle
arg_1_tensor = paddle.rand([1, 2, 3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = "NCHW"
res = paddle.nn.functional.prelu(arg_1,arg_2,data_format=arg_3,)

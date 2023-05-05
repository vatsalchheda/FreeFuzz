import paddle
arg_1_tensor = paddle.rand([2, 2, 1, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = True
arg_4 = "NCHW"
arg_5 = None
res = paddle.nn.functional.dropout2d(arg_1,p=arg_2,training=arg_3,data_format=arg_4,name=arg_5,)

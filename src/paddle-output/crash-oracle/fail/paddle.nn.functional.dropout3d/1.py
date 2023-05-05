import paddle
arg_1_tensor = paddle.rand([4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = "zeros"
arg_4 = "NCDHW"
arg_5 = None
res = paddle.nn.functional.dropout3d(arg_1,p=arg_2,training=arg_3,data_format=arg_4,name=arg_5,)

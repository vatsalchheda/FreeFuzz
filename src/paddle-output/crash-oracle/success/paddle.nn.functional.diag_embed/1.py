import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -47
arg_3 = 0
arg_4 = 2
res = paddle.nn.functional.diag_embed(arg_1,offset=arg_2,dim1=arg_3,dim2=arg_4,)

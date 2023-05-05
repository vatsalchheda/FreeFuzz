import paddle
arg_1_tensor = paddle.rand([4, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,16384,[4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
arg_4_tensor = paddle.rand([4, 3], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5_tensor = paddle.rand([4, 1], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
arg_6 = None
arg_7 = None
arg_8 = False
arg_9 = None
res = paddle.nn.functional.hsigmoid_loss(arg_1,arg_2,arg_3,arg_4,arg_5,path_table=arg_6,path_code=arg_7,is_sparse=arg_8,name=arg_9,)

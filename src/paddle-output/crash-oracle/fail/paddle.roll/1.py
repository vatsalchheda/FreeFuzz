import paddle
arg_1_tensor = paddle.rand([10], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,128,[1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = True
arg_3 = [arg_3_0,]
arg_4 = None
res = paddle.roll(arg_1,arg_2,arg_3,name=arg_4,)

import paddle
arg_1_tensor = paddle.randint(-32768,128,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
arg_3 = -16
arg_4 = -60
res = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)

import paddle
arg_1_tensor = paddle.randint(-128,4096,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = 0
arg_4 = -16
res = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)

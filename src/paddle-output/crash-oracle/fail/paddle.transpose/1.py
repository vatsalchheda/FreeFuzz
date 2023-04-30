import paddle
arg_1_tensor = paddle.randint(-4,256,[2, 3, 0], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2_2 = 2
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = paddle.transpose(arg_1,perm=arg_2,)

import paddle
arg_1_tensor = paddle.randint(-64,128,[18, 18], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.transpose(arg_1,perm=arg_2,)

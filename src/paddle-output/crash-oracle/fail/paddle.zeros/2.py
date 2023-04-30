import paddle
arg_1_0_tensor = paddle.randint(-128,4,[1], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2 = "float64"
res = paddle.zeros(arg_1,dtype=arg_2,)
